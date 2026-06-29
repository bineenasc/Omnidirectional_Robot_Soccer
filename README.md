# Deep Reinforcement Learning for 1v1 Robot Soccer with Heterogeneous Omni-Directional Robots

A reinforcement learning project in which two physically different three-wheel
omni-directional robots — **Viper** (small, fast) and **Titan** (large, slow) — learn to
navigate, control a ball, and score goals using **Proximal Policy Optimization (PPO)**.
Training runs in the **Webots R2025a** simulator, exposed to **Stable-Baselines3** through a
**Gymnasium** environment. A single shared policy controls both robots; their morphology is
encoded as an observation feature so the network can condition its behaviour on the active body.

**Institutional context:** Faculty of Engineering, University of Porto (FEUP) —
course *Topics in Intelligent Robotics*, programme M.IA, academic year 2025/2026.

> **Note — this is not a ROS project.** Although the generic submission template mentions ROS,
> the work runs entirely on **Webots + Python + Stable-Baselines3**. There are no ROS packages,
> no catkin/colcon build steps, and no makefile or compiled executable: the controllers are
> Python scripts launched automatically by Webots when a world is opened.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Requirements](#requirements)
4. [Project Layout](#project-layout)
5. [How to Run](#how-to-run)
6. [Method](#method)
7. [Results](#results)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [License](#license)

---

## Overview

The project is organised in two training stages:

- **Stage 1 — 1v0:** a single robot learns the foundational skills of locomotion, ball approach,
  and shooting, without an opponent.
- **Stage 2 — 1v1:** the 1v0 policy is fine-tuned in an adversarial setting. The observation
  space is extended from 24 to 30 dimensions (opponent position, velocity, and a blocking flag)
  by copying the existing weights and zero-initialising the new inputs ("weight surgery").

```
   STAGE 1 - 1v0                  weight surgery (24 -> 30)            STAGE 2 - 1v1
   locomotion, reach ball,   ------------------------------------>    adversarial play
   shoot at the goal              (1v0 policy fine-tuned)             against the other robot
```

---

## System Architecture

The Webots **Supervisor** doubles as the Gymnasium environment. It has omniscient access to all
node positions (read directly from the Webots API, so GPS-equivalent data carries no noise) and
drives the simulation step by step. Only LiDAR data is transmitted over inter-process
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

Each simulation step corresponds to 5 Webots basic timesteps of 8 ms, yielding a 40 ms control
period (25 Hz). Training runs in `SIMULATION_MODE_FAST`, decoupled from real time, to maximise
sample throughput.

---

## Requirements

| Component | Version |
|---|---|
| Operating system | Windows 10/11 (developed on); also runs on Linux and macOS |
| Webots | R2025a |
| Python | 3.11 or newer (tested on 3.11 and 3.13) |
| gymnasium | >= 0.29.0 |
| stable-baselines3[extra] | >= 2.3.0 |
| torch | >= 2.0.0 |
| numpy | >= 1.24.0 |
| matplotlib | >= 3.7.0 |
| pandas | >= 2.0.0 |

Install the Python dependencies into the same interpreter Webots is configured to use:

```bash
pip install -r requirements.txt
```

In Webots, set the interpreter under **Tools -> Preferences -> Python command** (for example
`python3` on Linux/macOS, or `py -3.13` on Windows) so the controllers use the environment where
the packages are installed.

### Robot platforms

| Parameter | Viper | Titan |
|---|---|---|
| Body radius | 12 cm | 16.5 cm |
| Approx. top speed | 0.45 m/s | 0.315 m/s |
| Wheel radius | 3.0 cm | 3.5 cm |
| Drive | 3 omni wheels at 120 degrees | 3 omni wheels at 120 degrees |
| Sensors | 1440-ray LiDAR, GPS, IMU | 1440-ray LiDAR, GPS, IMU |

---

## Project Layout

```
Omnidirectional_Robot_Soccer/
├── controllers/
│   ├── shared_configs.py              # field/ball/IPC/robot constants (single source of truth)
│   ├── robot_controller/
│   │   └── robot_controller.py        # omnidirectional inverse kinematics + LiDAR over IPC
│   ├── soccer_supervisor/             # ---------- STAGE 1 - 1v0 ----------
│   │   ├── soccer_supervisor.py       # Gymnasium environment + Webots Supervisor (entry point)
│   │   ├── train.py                   # PPO training loop (warm-start + curriculum)
│   │   ├── eval.py                    # evaluation (phase sweep + random-agent baseline)
│   │   ├── checkpoints/               # saved models (and backup_1v0/ originals)
│   │   ├── logs/                      # TensorBoard logs + episode_log.csv
│   │   └── plots/                     # training curves (PNG)
│   └── soccer_supervisor_1v1/         # ---------- STAGE 2 - 1v1 ----------
│       ├── soccer_supervisor_1v1.py   # 1v1 Gymnasium environment + Supervisor
│       ├── train_1v1.py               # warmup -> phase1 -> phase2 -> phase3 (self-play)
│       └── eval_1v1.py                # 1v1 match evaluation (Viper vs Titan)
├── protos/
│   ├── Viper.proto                    # fast/small robot definition
│   └── Titan.proto                    # slow/large robot definition
├── worlds/
│   ├── soccer.wbt                     # 1v0 world
│   └── soccer_1v1.wbt                 # 1v1 world
├── monitor_viper.py                   # live training monitor (Viper)
├── monitor_titan.py                   # live training monitor (Titan)
├── monitor_1v1.py                     # live training monitor (1v1)
├── requirements.txt
├── FIXES_1x0_README.md                # notes on the curriculum / eval / warm-start fixes
└── README.md
```

Main source files to read first: the RL environment, observation, reward and curriculum logic in
`controllers/soccer_supervisor/soccer_supervisor.py`; the 1v0 training loop in
`controllers/soccer_supervisor/train.py`; the evaluation in
`controllers/soccer_supervisor/eval.py`; and the 1v1 pipeline in
`controllers/soccer_supervisor_1v1/`.

---

## How to Run

The run mode is selected through the `MODE` flag in `controllers/shared_configs.py`
(`"train"` or `"eval"`).

### Train the 1v0 stage

1. Set `MODE = "train"` in `controllers/shared_configs.py`.
2. Open `worlds/soccer.wbt` in Webots and press **Run** (do not pause). Training starts
   automatically — the supervisor controller drives the loop. Disable 3D rendering with `Ctrl+4`
   to speed it up.
3. (Optional) Open a live monitor in a separate terminal: `python monitor_viper.py`.

Outputs are written to `controllers/soccer_supervisor/{checkpoints,logs,plots}/`. The final
models are `final_model_viper.zip` and `final_model_titan.zip` (each with a matching
`_vecnorm.pkl`).

### Evaluate the 1v0 stage

1. Set `MODE = "eval"`. 2. Open `worlds/soccer.wbt` and press **Run**.

The evaluation runs with the curriculum **frozen** so the difficulty cannot drift. With
`EVAL_PHASE = None` it sweeps phases 0 to 7 and prints a goal / own-goal / out / timeout table,
followed by a random-agent baseline. Set `EVAL_FAST = False` to watch the episodes in real time.
The model under test is selected with the `MODEL_PATH` constant.

### Train or evaluate the 1v1 stage

Use the controllers in `controllers/soccer_supervisor_1v1/` together with
`worlds/soccer_1v1.wbt`. `train_1v1.py` loads the 1v0 final models and extends the observation
from 24 to 30 dimensions; `eval_1v1.py` runs Viper-versus-Titan matches and reports wins and
draws.

---

## Method

| Element | Description |
|---|---|
| Observation (1v0) | 24-D, in the robot's local frame: ball distance and direction, attack-goal distance and direction, four goal posts (distance and direction), robot velocity, ball velocity. The 1v1 stage adds six opponent features for a 30-D vector. |
| Action | `Box(3,)` in `[-1, 1]`, mapped to `[vx, vz, omega]` and scaled to +-0.5 m/s and +-3.0 rad/s. |
| Reward | +300 goal, -200 own-goal, -25 ball out of bounds; ball-to-goal progress x15 (clamped with `max(0, delta)`); robot-to-ball progress x4; robot-ball-goal alignment x0.3; ball velocity toward goal up to +-1; anti-stall and post-stuck penalties; -0.003 per-step time penalty. |
| Curriculum (1v0) | Eight phases. The ball spawns progressively farther from the goal until phase 7 uses fully random positions. A phase is promoted when the goal rate reaches 40% over a 30-episode window, and demoted below 10% so the agent can re-consolidate. |
| PPO setup | MLP policy `[256, 256]`; `n_steps = 2048`; `batch_size = 256`; `n_epochs = 10`; gamma = 0.99; GAE lambda = 0.95; clip range 0.2; entropy coefficient 0.03; learning rate linearly decaying from 3e-4 to 1e-5. Rewards are normalised with `VecNormalize`. |

---

## Results

The results below are reported as measured within the available training budget, including a
degenerate behaviour that is presented as a finding because it directly motivates the reward
design discussion.

### 1v0 (deterministic evaluation, curriculum frozen)

| Metric | Value |
|---|---|
| Goal rate | 37.5% (3/8) |
| Timeout rate | 62.5% (5/8) |
| Own-goal rate | 0% |
| Ball-out rate | 0% |
| Mean episode reward | 134.05 +- 139.46 |
| Mean episode length | 189.6 +- 93.4 steps |
| Mean final ball-to-goal distance | 0.283 +- 0.120 m |
| Fastest goal | 55 steps (reward +320.77) |

The agent scores reliably at short and medium range and never concedes own-goals or sends the
ball out of bounds, but a large share of episodes end in timeout: it brings the ball close to the
goal (about 0.28 m) yet does not finish. Qualitatively, the deterministic policy tends to move to
a fixed point near a goal post and stall there instead of driving the ball across the line — a
local optimum of the dense reward (discussed below).

### 1v1 (20 evaluation episodes, self-play phase)

All matches end in draws, with neither robot scoring. This is consistent with the 1v0 finishing
weakness, compounded by the opponent obstructing the goal mouth.

### Engineering fixes delivered

Details are documented in `FIXES_1x0_README.md`. In summary: the original curriculum promotion
threshold (50% over 40 episodes) was effectively unreachable and kept training concentrated on the
trivial early phases; it was relaxed to 40% over 30 episodes with an added demotion rule (verified
to climb to phase 6 and self-correct). The evaluation protocol previously measured only the
trivial phase 0; it now sweeps all phases and compares against a random-agent baseline. The
training script also supports warm-start, allowing existing models to be refined instead of
retrained from scratch.

---

## Limitations and Future Work

- **Reward exploitation ("park near post").** The anti-stall penalties are too weak relative to
  the dense shaping, so a static position near the goal mouth is locally rewarding. The priority
  fix is an explicit anti-park penalty combined with a finishing bonus proportional to the ball
  speed across the goal line.
- **Limited training time** (a stated project constraint). Long-range finishing and 1v1 scoring
  require substantially more training steps and true simultaneous self-play.
- **Simulation only**, with simplified physics and no real-robot deployment.
- **Planned extensions:** dynamic-ball interception (the ball arrives with an initial velocity),
  a 2v2 setting with inter-agent communication, and a comparison of PPO against SAC and DDPG on
  the same environment.

---

## License

Released under the MIT License. See the `LICENSE` file for details.
