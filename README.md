# Deep Reinforcement Learning for 1v1 Robot Soccer with Heterogeneous Omni-Directional Robots

Two physically different three-wheel omni-directional robots — **Viper** (small, fast) and
**Titan** (large, slow) — learn to navigate, control a ball, and score goals with **Proximal
Policy Optimization (PPO)**. Training runs in **Webots R2025a**, exposed to **Stable-Baselines3**
through a **Gymnasium** environment. A *single* policy controls both robots: morphology is encoded
as an observation feature, so one network conditions its behaviour on the active body instead of
maintaining two separate models. Learning is staged with a competency-based **curriculum**: a
single-agent stage (1v0) for the foundational skills, followed by an adversarial stage (1v1) that
fine-tunes the policy against an opponent.

**Institutional context.** Faculty of Engineering, University of Porto (FEUP) — course
*Topics in Intelligent Robotics*, programme M.IA, academic year 2025/2026.

> **This is not a ROS project.** Despite the generic submission template mentioning ROS, the work
> runs entirely on **Webots + Python + Stable-Baselines3**. There are no ROS packages, no
> catkin/colcon build, and no makefile or compiled executable — the controllers are Python scripts
> launched automatically by Webots when a world is opened.

---

## Table of Contents

1. [Authors](#authors)
2. [Problem Formulation](#problem-formulation)
3. [System Architecture](#system-architecture)
4. [Requirements & Installation](#requirements--installation)
5. [Project Layout](#project-layout)
6. [How to Run](#how-to-run)
7. [Method](#method)
8. [Reproducibility](#reproducibility)
9. [Results](#results)
10. [Discussion: the "park-near-post" failure mode](#discussion-the-park-near-post-failure-mode)
11. [Limitations & Future Work](#limitations--future-work)
12. [References](#references)
13. [License](#license)

---

## Authors

| Author | Student № |
|---|---|
| Daniela Osório | up202208679 |
| Diogo Ferreira | up202205295 |
| Marcella Duque | up202512342 |
| Pabline Nascimento | up202512028 |

Repository: `https://github.com/bineenasc/Omnidirectional_Robot_Soccer` (private; access shared
with `asousa@fe.up.pt`).

---

## Problem Formulation

The task is modelled as a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ solved
with PPO. The agent observes a **privileged, low-dimensional state** built by the supervisor
(no raw pixels/LiDAR in the policy) and outputs **continuous body-frame velocities**.

- **State** $s \in \mathbb{R}^{24}$ (1v0) / $\mathbb{R}^{30}$ (1v1), all directions in the robot's
  local frame (see [Observation](#observation-space)).
- **Action** $a \in [-1,1]^3 \rightarrow [v_x, v_z, \omega]$, scaled to $\pm0.5$ m/s and
  $\pm3.0$ rad/s, decoded into wheel speeds by the omni-directional inverse kinematics.
- **Reward** $R(s,a,s')$: dense shaping + sparse terminal events (see [Reward](#reward-function)).
- **Discount** $\gamma = 0.99$; **control rate** 25 Hz (40 ms/step).
- **Episode termination:** goal scored, own goal, ball out of bounds, or step budget reached.

The central design hypothesis is that a **robot-local observation representation** lets a single
policy generalise across morphologies (mass, geometry, top speed) without learning an implicit
coordinate transform.

---

## System Architecture

The Webots **Supervisor** plays a dual role: it is the Gymnasium environment *and* the omniscient
simulation coordinator. It reads all node positions directly from the Webots API (noise-free,
GPS-equivalent) and drives the simulation step by step. Only LiDAR travels over inter-process
communication (IPC); because Webots R2025a decodes receiver messages as UTF-8, binary packets are
Base64-encoded before transmission.

```
                       channel 0  ->  action [vx, vz, omega]
   +--------------+  ------------------------------------------>  +-------------------+
   |  Supervisor  |                                               |  Robot controller |
   | (omniscient  |                                               |  (omnidirectional |
   |  positions;  |  <------------------------------------------  |   inverse         |
   |  Gym env)    |          channel 1  <-  LiDAR (Base64)        |   kinematics)     |
   +--------------+                                               +-------------------+
```

Each RL step advances 5 Webots basic timesteps of 8 ms (40 ms control period, 25 Hz). Training
runs in `SIMULATION_MODE_FAST`, decoupled from real time, to maximise sample throughput.

---

## Requirements & Installation

| Component | Version |
|---|---|
| Operating system | Windows 10/11 (developed on); also Linux / macOS |
| Webots | R2025a |
| Python | 3.11+ (tested on 3.11 and 3.13) |
| gymnasium | >= 0.29.0 |
| stable-baselines3[extra] | >= 2.3.0 |
| torch | >= 2.0.0 |
| numpy | >= 1.24.0 |
| matplotlib | >= 3.7.0 |
| pandas | >= 2.0.0 |

```bash
pip install -r requirements.txt
```

Install into the **same interpreter Webots uses**. In Webots, set it under
**Tools → Preferences → Python command** (e.g. `python3` on Linux/macOS, `py -3.13` on Windows).

### Robot platforms

| Parameter | Viper | Titan |
|---|---|---|
| Colour | red | blue |
| Body radius | 0.120 m | 0.165 m |
| Wheel radius | 0.030 m | 0.035 m |
| Wheel distance (centre→wheel) | 0.138 m | 0.170 m |
| Max wheel velocity | 15.0 rad/s | 9.0 rad/s |
| Approx. top speed | ~0.45 m/s | ~0.315 m/s |
| Drive | 3 omni wheels @ 120° | 3 omni wheels @ 120° |
| Sensors | 1440-ray LiDAR, GPS, IMU | 1440-ray LiDAR, GPS, IMU |

Field: 7.4 × 10.4 m (Webots NUE); goals 1.5 m wide at `z = ±4.55 m`; ball radius 0.025 m,
mass 0.055 kg.

---

## Project Layout

```
Omnidirectional_Robot_Soccer/
├── controllers/
│   ├── shared_configs.py              # field/ball/IPC/robot constants (single source of truth)
│   ├── robot_controller/
│   │   └── robot_controller.py        # omnidirectional inverse kinematics + LiDAR over IPC
│   ├── soccer_supervisor/             # ---------- STAGE 1 · 1v0 ----------
│   │   ├── soccer_supervisor.py       # Gymnasium env + Webots Supervisor (entry point)
│   │   ├── train.py                   # PPO training loop (warm-start + curriculum)
│   │   ├── eval.py                    # evaluation: phase sweep + random-agent baseline
│   │   ├── plot_results.py            # per-episode learning curve from episode_log.csv
│   │   ├── plot_stages.py             # per-stage reward/goal-rate bars
│   │   ├── plot_robot_curves.py       # per-epoch curves per robot (works mid-run)
│   │   ├── checkpoints/               # saved models (+ backup_1v0/ originals)
│   │   ├── logs/                      # TensorBoard logs + episode_log.csv
│   │   └── plots/                     # generated figures (PNG)
│   └── soccer_supervisor_1v1/         # ---------- STAGE 2 · 1v1 ----------
│       ├── soccer_supervisor_1v1.py   # 1v1 Gymnasium env + Supervisor
│       ├── train_1v1.py               # warmup → phase1 → phase2 → phase3 (self-play)
│       └── eval_1v1.py                # 1v1 match evaluation (Viper vs Titan)
├── protos/      Viper.proto · Titan.proto          # robot PROTO definitions
├── worlds/      soccer.wbt · soccer_1v1.wbt        # 1v0 and 1v1 worlds
├── monitor_viper.py · monitor_titan.py · monitor_1v1.py   # live training monitors
├── requirements.txt
├── FIXES_1x0_README.md                # notes on the curriculum / eval / warm-start fixes
└── README.md
```

**Read first:** `controllers/soccer_supervisor/soccer_supervisor.py` (environment, observation,
reward, curriculum), then `train.py` (1v0 loop), `eval.py` (evaluation), and the
`soccer_supervisor_1v1/` package (adversarial stage).

---

## How to Run

The run mode is the `MODE` flag in `controllers/shared_configs.py` (`"train"` or `"eval"`).

**Train (1v0).** Set `MODE = "train"`, open `worlds/soccer.wbt`, press **Run** (do not pause).
Training starts automatically; press `Ctrl+4` to disable rendering and speed it up. Outputs go to
`controllers/soccer_supervisor/{checkpoints,logs,plots}/`; finals are `final_model_viper.zip` and
`final_model_titan.zip` (+ `_vecnorm.pkl`). Optional live view: `py -3.13 monitor_viper.py`.

**Evaluate (1v0).** Set `MODE = "eval"`, open `worlds/soccer.wbt`, press **Run**. Evaluation runs
with the curriculum **frozen** so difficulty cannot drift. With `EVAL_PHASE = None` it sweeps
phases 0–7 and prints a goal / own-goal / out / **timeout** table, then a **random-agent
baseline**. `EVAL_FAST = False` watches in real time; `MODEL_PATH` selects the model.

**1v1 stage.** Use `controllers/soccer_supervisor_1v1/` with `worlds/soccer_1v1.wbt`.
`train_1v1.py` loads the 1v0 finals and extends the observation 24→30; `eval_1v1.py` runs
Viper-vs-Titan matches and reports wins/draws.

**Plots (no Webots needed).** From `controllers/soccer_supervisor/`:
`py -3.13 plot_results.py` (learning curve), `py -3.13 plot_stages.py` (per-stage bars),
`py -3.13 plot_robot_curves.py` (per-epoch curves). All read `logs/episode_log.csv`.

---

## Method

### Observation space

24-D in 1v0 (30-D in 1v1), all directions rotated into the robot's local frame via
`theta = atan2(-m1, m0)` (yaw from the orientation matrix), so that a positive local-z
component means "ahead".

| Index | Feature | Range |
|---|---|---|
| 0 | Robot type ID (0 = Viper, 1 = Titan) | {0,1} |
| 1 | Distance to ball | [0,1] |
| 2–3 | Direction to ball (x,z) | [-1,1]² |
| 4 | Distance to attack goal | [0,1] |
| 5–6 | Direction to attack goal (x,z) | [-1,1]² |
| 7–18 | 4 goal posts × (distance, dir_x, dir_z) | [0,1]×[-1,1]² |
| 19–21 | Robot velocity (v_x, v_z, ω), local | [-1,1]³ |
| 22–23 | Ball velocity (v_x, v_z), local | [-1,1]² |
| 24–29 | *(1v1 only)* opponent distance, direction, velocity, blocking flag | mixed |

### Action space

$a \in [-1,1]^3 \rightarrow v_x = a_0 \cdot 0.5$ m/s, $v_z = a_1 \cdot 0.5$ m/s,
$\omega = a_2 \cdot 3.0$ rad/s. Body velocities are decoded into the three wheel speeds by the
omni-directional inverse kinematics in the robot controller and clamped to each robot's maximum.

### Reward function

| Component | Value | Purpose |
|---|---|---|
| Goal scored | **+300** (terminal) | primary objective |
| Own goal | **−200** (terminal) | avoid self-scoring |
| Ball out of bounds | **−25** (terminal) | keep play active |
| Ball → goal progress | `+15 · max(0, Δd)` | dominant dense signal; clamp prevents ball-fleeing |
| Robot → ball progress | `+4 · Δd` | learn to reach the ball |
| Robot–ball–goal alignment | `+0.3 · cos θ` | position behind the ball |
| Ball velocity toward goal | up to ±1.0 | reward useful ball motion |
| Contact bonus | +0.25 directed / +0.05 otherwise | encourage purposeful touches |
| Stillness penalty | −0.10/step after 20 motionless steps | anti-stall |
| Post-stuck penalty | −0.30/step after 20 steps near a post | escape deadlocks |
| Time penalty | −0.003/step | encourage faster completion |

The 1v1 reward replaces the own-goal term with an **opponent-goal penalty (−250)** and adds an
**opponent-proximity penalty (−0.08)** within 0.30 m.

### Curriculum (1v0)

Eight phases with growing ball-to-goal distance and step budget:

| Phase | Ball distance to goal | Max steps |
|---|---|---|
| 0 | 0.05–0.20 m (robot pre-aligned) | 350 |
| 1 | 0.20–0.50 m | 500 |
| 2 | 0.50–1.00 m | 650 |
| 3 | 1.00–1.75 m | 800 |
| 4 | 1.75–2.75 m | 950 |
| 5 | 2.75–4.00 m | 1100 |
| 6 | attacking half (z > 0) | 1200 |
| 7 | fully random (full field) | 1250 |

Promotion when goal rate ≥ **40 %** over a 30-episode window; **demotion** below **10 %** so the
agent can re-consolidate. *(An initial 50 %/40-episode criterion proved unreachable and stalled
the curriculum on the trivial phases; the revision unblocks progression up to phase 6.)*

The **1v1** curriculum has four phases: *warmup* and *phase1* (static opponent, near-1v0 regime),
*phase2* (scripted go-to-ball/shoot opponent), and *phase3* (frozen self-play snapshot refreshed
each epoch, fully random spawns).

### PPO setup

| Hyperparameter | Value |
|---|---|
| Policy | MLP `[256, 256]`, shared across robots |
| `n_steps` | 2048 |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `gamma` (γ) | 0.99 |
| GAE `lambda` (λ) | 0.95 |
| Clip range | 0.2 |
| Entropy coef. | 0.03 |
| Learning rate | linear 3e-4 → 1e-5 |
| Reward normalisation | `VecNormalize` (clip 50) |

The 1v0 stage uses a hot-swap mechanism to expose one policy to both bodies; the 1v1 stage
alternates the active learner while the other robot acts as the (scripted or frozen) opponent.
Training supports **warm-start** (`CONTINUE_FROM_FINAL`): continue from `final_model_*.zip`
instead of training from scratch, with the original models automatically backed up to
`checkpoints/backup_1v0/`.

---

## Reproducibility

- **Determinism.** Evaluation uses deterministic actions and a **frozen curriculum** (fixed phase)
  so spawn distributions and termination conditions are identical across runs.
- **Baseline.** `eval.py` also evaluates a **random-action agent** on the hardest phase, giving a
  reference point for every metric.
- **Logging.** Every episode is appended to `logs/episode_log.csv`
  (`episode, timestep, reward, length, goal, stage, robot`); TensorBoard event files are saved
  per robot under `logs/`.
- **Artifacts.** Per-epoch checkpoints (`epoch_NN_<robot>.zip`) and matching `VecNormalize`
  statistics (`*_vecnorm.pkl`) are saved so any intermediate policy can be reloaded and evaluated.
- **Figures.** All plots regenerate from `episode_log.csv` via the `plot_*.py` scripts; the
  readers tolerate partial writes (the CSV may be written while training runs).

---

## Results

Reported as measured within the available training budget. We include a degenerate behaviour as a
**finding**, because it directly motivates the reward-design discussion.

### 1v0 — deterministic evaluation, curriculum frozen

| Metric | Value |
|---|---|
| Goal rate | 37.5 % (3/8) |
| Timeout rate | 62.5 % (5/8) |
| Own-goal rate | 0 % |
| Ball-out rate | 0 % |
| Mean episode reward | 134.05 ± 139.46 |
| Mean episode length | 189.6 ± 93.4 steps |
| Mean final ball-to-goal distance | 0.283 ± 0.120 m |
| Fastest goal | 55 steps (reward +320.77) |

The agent scores reliably at short/medium range and never concedes own-goals or sends the ball
out, but most episodes end in **timeout**: it brings the ball close (~0.28 m) yet does not finish.
Across training, **mean reward stays high (~200) while the goal rate declines (~80 % → ~25 %)** as
the curriculum hardens — a clear separation between optimised shaping and the true objective.

### 1v1 — 20 evaluation episodes (self-play phase)

| Outcome | Count |
|---|---|
| Viper wins | 0 |
| Titan wins | 0 |
| **Draws** | **20** |

No goals by either robot — consistent with the 1v0 finishing weakness compounded by the opponent
obstructing the goal mouth.

### Engineering fixes delivered

See `FIXES_1x0_README.md`. Summary: (i) the curriculum promotion threshold was relaxed
(50 %/40 ep → 40 %/30 ep) and a **demotion** rule added, unblocking progression (verified to climb
to phase 6); (ii) the evaluation now **sweeps all phases** and adds a **random-agent baseline**
(previously it silently measured only the trivial phase 0); (iii) training gained **warm-start**
with automatic model backup.

---

## Discussion: the "park-near-post" failure mode

The dominant failure is a textbook **reward-shaping local optimum**. The dense terms (alignment +
post-relative features) make a static position near the goal mouth locally rewarding, while the
anti-stall penalties (−0.10/−0.30 per step) are too small to overcome it and the time penalty
(−0.003) is negligible. The agent therefore maximises accumulated reward by **parking near a goal
post** rather than completing the shot — which is exactly why reward stays high while the goal rate
and the timeout rate both worsen. Both Viper and Titan exhibit the same behaviour, indicating the
bottleneck is the **reward balance**, not a specific embodiment. The principled fix is a
potential-based finishing term (a bonus proportional to ball speed across the goal line) plus a
stronger explicit anti-park penalty, combined with a larger training budget.

---

## Limitations & Future Work

- **Reward exploitation** ("park near post") — re-balance shaping with an explicit anti-park
  penalty and a finishing bonus.
- **Limited training time** (stated project constraint) — long-range finishing and 1v1 scoring
  need substantially more steps and **true simultaneous self-play**.
- **Simulation only** — simplified physics, no real-robot transfer; the policy ignores LiDAR
  (analytical GPS features only), so obstacle awareness is absent.
- **Planned extensions** — dynamic-ball interception (ball arrives with initial velocity), a 2v2
  setting with inter-agent communication, and an algorithmic comparison (PPO vs SAC vs DDPG).

---

## References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Raffin et al., *Stable-Baselines3: Reliable RL Implementations*, JMLR 2021.
- Bengio et al., *Curriculum Learning*, ICML 2009.
- Kitano et al., *RoboCup: A Challenge Problem for AI*, 1997.
- Cyberbotics, *Webots: Professional Mobile Robot Simulation*.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
