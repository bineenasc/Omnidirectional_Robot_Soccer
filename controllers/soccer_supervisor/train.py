"""
Training script for the 1×0 soccer RL agent.

Called by soccer_supervisor.py (the Webots controller) so it never imports
that module directly — avoiding circular imports.  The raw SoccerEnv instance
is created in soccer_supervisor.py and passed here as a parameter.

Epoch loop (per robot)
──────────────────────
  Each robot (Viper, Titan) is trained independently with its own PPO model
  and VecNormalize statistics, avoiding catastrophic forgetting from alternation.

  Viper  → N_EPOCHS epochs → final_model_viper.zip
  Titan  → N_EPOCHS epochs → final_model_titan.zip

Outputs (relative to this file's directory)
────────────────────────────────────────────
  checkpoints/epoch_NN_<robot>.zip          — periodic snapshots
  checkpoints/final_model_<robot>.zip       — final model per robot
  logs/                                     — TensorBoard event files
  plots/training_curves_<robot>.png         — reward + goal-rate per robot
"""

from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_EPOCHS        = 8               # epochs per robot — com warm-start 8 bastam p/ refinar
STEPS_PER_EPOCH = 100_000         # steps per epoch (~25 min cada → cabe num dia)
ROBOT_SEQUENCE  = ["viper", "titan"]

# [FIX-RESULTADO-HOJE] Continuar a partir do final_model_<robot>.zip existente
# em vez de recomeçar do zero. Os modelos já dominam as fases fáceis (60% no
# eval trivial); com o currículo destravado eles avançam para fases difíceis
# muito mais rápido do que treinar de novo do zero (poupa ~11 h).
# Coloque False para treinar de raiz.
CONTINUE_FROM_FINAL = True

# [CRITICO-2 fix] Rede simétrica — value function aprende melhor
POLICY_KWARGS: dict = dict(
    net_arch = [256, 256],
)

# [CRITICO-3 fix] Learning rate com decaimento linear 3e-4 → 1e-5
def _lr_schedule(progress_remaining: float) -> float:
    """Linear decay: 3e-4 no início, 1e-5 no fim."""
    return max(1e-5, 3e-4 * progress_remaining)

PPO_KWARGS: dict = dict(
    policy        = "MlpPolicy",
    # [FIX] era 1000 (= 1 episódio). 2048 (default SB3) recolhe vários episódios
    # por update → estimativas de vantagem (GAE) muito mais estáveis. Como o
    # _max_steps varia 350–1250 por fase, alinhar n_steps a 1 episódio era frágil.
    n_steps       = 2048,
    batch_size    = 256,          # 2048 / 256 = 8 minibatches por época
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.03,         # [MEDIO-3 fix] era 0.01; 0.03 evita entropy collapse
    learning_rate = _lr_schedule,
    verbose       = 1,
)

_HERE        = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR    = os.path.join(_HERE, "checkpoints")
_LOG_DIR     = os.path.join(_HERE, "logs")
_PLOT_DIR    = os.path.join(_HERE, "plots")
EPISODE_CSV  = os.path.join(_LOG_DIR, "episode_log.csv")


# ── CSV episode logger (feeds monitor_viper/titan and plot_stages) ────────────
# Columns: episode, timestep, reward, length, goal, stage, robot
# "stage" = global epoch index: viper uses 0..N-1, titan uses N..2N-1.

def _init_episode_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["episode", "timestep", "reward", "length", "goal", "stage", "robot"]
        )


class _EpisodeCSVCallback(BaseCallback):
    """Appends one row to the CSV at the end of each episode."""

    def __init__(self, csv_path: str, stage: int, robot: str) -> None:
        super().__init__(verbose=0)
        self.csv_path = csv_path
        self.stage    = stage
        self.robot    = robot
        self._ep      = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep += 1
                goal = 1 if info.get("goal_scored") else 0
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self._ep, self.num_timesteps, f"{ep['r']:.4f}",
                         ep["l"], goal, self.stage, self.robot]
                    )
        return True


# ── Main entry point ──────────────────────────────────────────────────────────

def train(env_raw) -> None:
    """
    Train independent PPO models for Viper and Titan sequentially.
    Each robot gets its own model and VecNormalize statistics — no cross-robot
    interference / catastrophic forgetting.
    """
    for d in (_CKPT_DIR, _LOG_DIR, _PLOT_DIR):
        os.makedirs(d, exist_ok=True)
    _init_episode_csv(EPISODE_CSV)
    _backup_existing_finals()  # protege os modelos já treinados antes de continuar

    # Train Viper first (already in world at startup)
    print("\n" + "═" * 60)
    print("  TRAINING: VIPER")
    print("═" * 60)
    env_raw.set_robot_type("viper")
    env_raw._curriculum_step = 0
    env_raw._phase = 0
    env_raw._goal_history.clear()
    _train_robot(env_raw, "viper", stage0=0)

    # Swap to Titan and train
    print("\n" + "═" * 60)
    print("  TRAINING: TITAN")
    print("═" * 60)
    env_raw.swap_robot("titan")
    env_raw._curriculum_step = 0
    env_raw._phase = 0
    env_raw._goal_history.clear()
    _train_robot(env_raw, "titan", stage0=N_EPOCHS)

    print("\n[train] All robots trained. Models saved to checkpoints/")


def _backup_existing_finals() -> None:
    """Copy any existing final_model_*.zip/pkl to checkpoints/backup_1v0/ once.

    Warm-start overwrites the final models at the end of each run; this keeps a
    pristine copy of the originally-trained 1v0 models so they're never lost.
    """
    import shutil
    backup = os.path.join(_CKPT_DIR, "backup_1v0")
    os.makedirs(backup, exist_ok=True)
    for robot in ROBOT_SEQUENCE:
        for suffix in (".zip", "_vecnorm.pkl"):
            src = os.path.join(_CKPT_DIR, f"final_model_{robot}{suffix}")
            dst = os.path.join(backup, f"final_model_{robot}{suffix}")
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)
                print(f"[backup] {src} → {dst}")


def _train_robot(env_raw, robot_name: str, stage0: int = 0) -> PPO:
    """
    Train one robot for N_EPOCHS epochs with its own fresh model and VecNormalize.
    ``stage0`` = global epoch offset for the CSV (viper: 0, titan: N_EPOCHS).
    """
    ckpt_dir = _CKPT_DIR
    log_dir  = _LOG_DIR

    # ── Wrapper stack (fresh for each robot) ─────────────────────────────────
    env     = Monitor(env_raw)
    vec_env = DummyVecEnv([lambda: env])

    final_stem    = os.path.join(ckpt_dir, f"final_model_{robot_name}")
    vecnorm_path  = final_stem + "_vecnorm.pkl"
    model_path    = final_stem + ".zip"
    warm_start    = CONTINUE_FROM_FINAL and os.path.isfile(model_path)

    if warm_start and os.path.isfile(vecnorm_path):
        # Reaproveita as estatísticas de normalização da reward já aprendidas.
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs    = False,
            norm_reward = True,
            clip_reward = 50.0,  # [MEDIO fix] era 10.0; 50 preserva sinal do gol (+300)
            gamma       = PPO_KWARGS["gamma"],
        )

    if warm_start:
        print(f"[{robot_name}] WARM-START: a continuar de {model_path}")
        model = PPO.load(
            model_path,
            env             = vec_env,
            tensorboard_log = log_dir,
            custom_objects  = {"learning_rate": _lr_schedule,
                               "lr_schedule":   _lr_schedule,
                               "n_steps":       PPO_KWARGS["n_steps"],
                               "batch_size":    PPO_KWARGS["batch_size"],
                               "ent_coef":      PPO_KWARGS["ent_coef"]},
        )
    else:
        print(f"[{robot_name}] A treinar de raiz (sem warm-start).")
        model = PPO(
            env             = vec_env,
            policy_kwargs   = POLICY_KWARGS,
            tensorboard_log = log_dir,
            **PPO_KWARGS,
        )

    epoch_rewards:    list[float] = []
    epoch_goal_rates: list[float] = []

    for epoch in range(N_EPOCHS):
        print(f"\n[{robot_name}] Epoch {epoch + 1}/{N_EPOCHS}  "
              f"steps_so_far={model.num_timesteps}")

        stats_cb = _StatsCallback()
        csv_cb   = _EpisodeCSVCallback(EPISODE_CSV, stage=stage0 + epoch,
                                       robot=robot_name)
        try:
            model.learn(
                total_timesteps     = STEPS_PER_EPOCH,
                reset_num_timesteps = False,
                callback            = CallbackList([stats_cb, csv_cb]),
                tb_log_name         = f"ppo_{robot_name}",
                progress_bar        = True,
            )
        finally:
            # Crash-safe mid-epoch vecnorm snapshot
            vec_env.save(os.path.join(ckpt_dir,
                f"epoch_{epoch:02d}_{robot_name}_vecnorm.pkl"))

        stem = os.path.join(ckpt_dir, f"epoch_{epoch:02d}_{robot_name}")
        model.save(stem)
        vec_env.save(stem + "_vecnorm.pkl")

        if stats_cb.ep_rewards:
            mean_r  = float(np.mean(stats_cb.ep_rewards))
            goal_rt = float(np.mean(stats_cb.ep_goals))
        else:
            mean_r, goal_rt = 0.0, 0.0

        epoch_rewards.append(mean_r)
        epoch_goal_rates.append(goal_rt)
        print(f"  episodes={len(stats_cb.ep_rewards)}"
              f"  mean_reward={mean_r:.3f}"
              f"  goal_rate={goal_rt:.1%}")

    # ── Save final model for this robot ──────────────────────────────────────
    final_stem = os.path.join(ckpt_dir, f"final_model_{robot_name}")
    model.save(final_stem)
    vec_env.save(final_stem + "_vecnorm.pkl")
    print(f"\n[{robot_name}] Done. Final model → {final_stem}.zip")

    _plot_curves(epoch_rewards, epoch_goal_rates, robot_name)
    return model


# ── Episode-stats callback ────────────────────────────────────────────────────

class _StatsCallback(BaseCallback):
    """Collects per-episode reward and goal-rate within a single learn() call."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.ep_rewards: list[float] = []
        self.ep_goals:   list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(ep["r"])
                self.ep_goals.append(1.0 if info.get("goal_scored") else 0.0)
        return True


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_curves(
    rewards:    list[float],
    goal_rates: list[float],
    robot_name: str,
) -> None:
    if not rewards:
        return

    color  = "#2196F3" if robot_name == "viper" else "#F44336"
    epochs = list(range(1, len(rewards) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Training curves — {robot_name.capitalize()}",
                 fontweight="bold")

    for i, r in enumerate(rewards):
        ax1.bar(epochs[i], r, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Mean episode reward")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.grid(axis="y", alpha=0.3)

    for i, g in enumerate(goal_rates):
        ax2.bar(epochs[i], g * 100.0, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Goal rate (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(_PLOT_DIR, f"training_curves_{robot_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[{robot_name}] Plot saved → {out}")
