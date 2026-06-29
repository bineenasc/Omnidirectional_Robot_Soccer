# ------------------------------- #
# EVALUATE/COMPARE TRAINED MODELS #
# ------------------------------- #

''' 
Metrics reported
 -> Goal rate (goals scored / total episodes)
 -> Own-goal rate
 -> Ball-out rate
 -> Mean / std reward per episode
 -> Mean / std final distance of ball to attack goal (metres)
 -> Mean / std episode length (steps)
 -> Per-episode breakdown table
'''

# ------- #
# IMPORTS #
# ------- #

from __future__ import annotations
 
import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from soccer_supervisor import SoccerEnv
from shared_configs import FIELD


#  Allow running as both a Webots controller and a plain Python script 
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))   # for shared_configs
 
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
MODEL_PATH = os.path.join(_CKPT_DIR, "final_model_viper")
N_EPISODES = 20
DETERMINISTIC = True # False = stochastic actions (more variety in eval)

# [FIX] Fase do currículo onde se avalia.
#   7  = spawn totalmente aleatório / campo todo  → métrica HONESTA de skill.
#   0  = bola colada ao gol (trivial)             → o eval antigo media isto.
# Defina EVAL_PHASE = None para varrer todas as fases (0..7) — recomendado.
EVAL_PHASE: int | None = None   # None → sweep 0..7

# Avaliar também um agente aleatório como baseline (exigido no documento).
RUN_RANDOM_BASELINE = True

# Episódios por fase durante o sweep (mantém baixo: eval pode correr em FAST).
SWEEP_EPISODES_PER_PHASE = 8

# FAST = mede rápido (recomendado para métrica). False = real-time (ver o jogo).
EVAL_FAST = True

# [COMPARA] Comparar o modelo atual vs o backup pré-warm-start, numa só corrida.
# Põe COMPARE_WITH_BACKUP = True e MODE = "eval" → imprime tabela lado-a-lado
# nas fases COMPARE_PHASES, e escolhes o melhor para o artigo/vídeo.
COMPARE_WITH_BACKUP = False
COMPARE_PHASES      = [0, 3, 7]
COMPARE_EPISODES    = 8



# --------- #
# FUNCTIONS #
# --------- #

def _find_vecnorm(model_path: str) -> str | None:
    """
    Given  .../checkpoints/epoch_05_viper.zip
    return .../checkpoints/epoch_05_viper_vecnorm.pkl  if it exists.
    """
    stem = os.path.splitext(model_path)[0]
    pkl  = stem + "_vecnorm.pkl"
    return pkl if os.path.isfile(pkl) else None


# --- EVAL --- #

def _summarise(results: list[dict], n_episodes: int, label: str = "") -> dict:
    """Print and return aggregate metrics for a list of per-episode results."""
    rewards = np.array([r["reward"] for r in results])
    steps   = np.array([r["steps"] for r in results])
    dists   = np.array([r["dist_to_goal"] for r in results])
    n_goals     = sum(r["goal_scored"] for r in results)
    n_own_goals = sum(r["own_goal"] for r in results)
    n_ball_out  = sum(r["ball_out"] for r in results)
    n_timeout   = sum(r["timeout"] for r in results)

    if label:
        print(f"\n=== {label} ===")
    print(f"Avg reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Avg steps: {steps.mean():.1f} ± {steps.std():.1f}")
    print(f"Avg final dist to goal: {dists.mean():.3f} ± {dists.std():.3f} m")
    print(f"Goal rate:      {n_goals}/{n_episodes} = {n_goals/n_episodes:.1%}")
    print(f"Own-goal rate:  {n_own_goals}/{n_episodes} = {n_own_goals/n_episodes:.1%}")
    print(f"Ball-out rate:  {n_ball_out}/{n_episodes} = {n_ball_out/n_episodes:.1%}")
    print(f"Timeout rate:   {n_timeout}/{n_episodes} = {n_timeout/n_episodes:.1%}")
    return dict(
        goal_rate     = n_goals / n_episodes,
        own_goal_rate = n_own_goals / n_episodes,
        ball_out_rate = n_ball_out / n_episodes,
        timeout_rate  = n_timeout / n_episodes,
        reward_mean   = float(rewards.mean()),
        dist_mean     = float(dists.mean()),
    )


def model_evaluate(
        model_path: str,
        n_episodes: int,
        deterministic: bool,
        eval_phase: int | None = None,
        random_agent: bool = False,
    ) -> list[dict]:
    """
    Evaluate one model (or a random agent) on a FIXED curriculum phase.

    eval_phase: 0..7 to lock difficulty (7 = full-field random spawn = honest
                skill measure).  None falls back to the env's default (phase 0).
    random_agent: if True, ignore the model and sample uniform random actions
                  (baseline required by the project's objectives doc).
    """

    GOAL_Z = FIELD["goal_z_attack"]

    who = "RANDOM AGENT (baseline)" if random_agent else model_path
    print(f"Model: {who}")
    print(f"Episodes: {n_episodes}  |  Deterministic: {deterministic}"
          f"  |  Phase: {eval_phase if eval_phase is not None else 'env-default'}")
    print(f"{'─'*51}\n")

    env_raw  = SoccerEnv()
    # FAST → métrica rápida.  REAL_TIME → visualização suave do jogo.
    if EVAL_FAST:
        env_raw.simulationSetMode(env_raw.SIMULATION_MODE_FAST)
    else:
        env_raw.simulationSetMode(env_raw.SIMULATION_MODE_REAL_TIME)
    # [FIX] Lock the difficulty so the curriculum can't drift during eval.
    env_raw.freeze_curriculum(True)
    env_mon  = Monitor(env_raw)
    vec_env  = DummyVecEnv([lambda: env_mon])

    model = None
    if not random_agent:
        vecnorm_path = _find_vecnorm(model_path)
        if vecnorm_path:
            print(f" VecNorm: {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False   # freeze running stats during eval
            vec_env.norm_reward = False   # return raw rewards for reporting
        else:
            print(" VecNorm: not found — rewards will be raw (unnormalised)")
            vec_env = VecNormalize(
                vec_env, norm_obs=False, norm_reward=False, gamma=0.99,
            )
        model = PPO.load(model_path, env=vec_env)
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=False, norm_reward=False, gamma=0.99,
        )

    results = []

    for ep in range(n_episodes):
        # [FIX] Force the requested phase BEFORE each reset.
        if eval_phase is not None:
            env_raw.set_phase(eval_phase)

        obs = vec_env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False
        last_info = {}

        while not done:
            if random_agent:
                action = np.array([env_raw.action_space.sample()])
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1
            last_info = infos[0]
            done = bool(dones[0])

        # final ball dist to goal (read directly from the Webots node)
        ball_p = env_raw._ball_node.getPosition()
        ball_x = ball_p[0]
        ball_z = ball_p[2]
        dist_to_goal = math.hypot(ball_x, ball_z - GOAL_Z)

        goal     = bool(last_info.get("goal_scored", False))
        own_goal = bool(last_info.get("own_goal", False))
        ball_out = bool(last_info.get("ball_out", False))
        timeout  = not (goal or own_goal or ball_out)   # [FIX] ran out of steps

        results.append(
            dict(
                ep = ep + 1, reward = ep_reward, steps = ep_steps,
                goal_scored = goal, own_goal = own_goal, ball_out = ball_out,
                timeout = timeout, dist_to_goal = dist_to_goal,
                ball_x = ball_x, ball_z = ball_z,
            )
        )

        tag = ("GOAL" if goal else "OWN-GOAL" if own_goal
               else "OUT" if ball_out else "timeout")
        print(f"Episode {ep + 1:>2} | R={ep_reward:+8.2f}  steps={ep_steps:>4}"
              f"  dist={dist_to_goal:.3f} m  [{tag}]")

    vec_env.close()
    _summarise(results, n_episodes)
    return results


def evaluate_phase_sweep(
        model_path: str,
        episodes_per_phase: int,
        deterministic: bool,
    ) -> None:
    """Evaluate the model on each curriculum phase 0..7 and print a table."""
    rows = []
    for ph in range(8):
        print(f"\n{'='*60}\n PHASE {ph}\n{'='*60}")
        res = model_evaluate(model_path, episodes_per_phase, deterministic,
                             eval_phase=ph)
        s = _summarise(res, episodes_per_phase, label=f"Phase {ph} summary")
        rows.append((ph, s))

    print(f"\n{'='*60}\n SWEEP SUMMARY — {Path(model_path).stem}\n{'='*60}")
    print(f"{'phase':>5} | {'goal%':>6} | {'own%':>5} | {'out%':>5} | "
          f"{'timeout%':>8} | {'dist(m)':>7}")
    for ph, s in rows:
        print(f"{ph:>5} | {s['goal_rate']*100:>5.0f}% | "
              f"{s['own_goal_rate']*100:>4.0f}% | {s['ball_out_rate']*100:>4.0f}% | "
              f"{s['timeout_rate']*100:>7.0f}% | {s['dist_mean']:>7.3f}")
    overall = np.mean([s['goal_rate'] for _, s in rows])
    print(f"\nMean goal rate across phases: {overall:.1%}")

def model_compare(
        list_of_model_paths: list[str], 
        n_episodes: int, 
        deterministic: bool,
        cmap: str = "rainbow"
    ) -> None:

    # plots bar charts comparing multiple models across the above metrics
    # 1 subplot per metric

    # --- get evals of all models --- #
    summaries = []

    for model_path in list_of_model_paths:

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*70}\n")

        results = model_evaluate(
            model_path=model_path,
            n_episodes=n_episodes,
            deterministic=deterministic
        )

        # avg of model performance
        rewards = np.array([r["reward"] for r in results])
        steps = np.array([r["steps"] for r in results])
        dists = np.array([r["dist_to_goal"] for r in results])
        n_goals = sum(r["goal_scored"] for r in results)
        n_own_goals = sum(r["own_goal"] for r in results)
        n_ball_out = sum(r["ball_out"] for r in results)

        summaries.append(
            {
                "name": Path(model_path).stem,

                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),

                "steps_mean": steps.mean(),
                "steps_std": steps.std(),

                "dist_mean": dists.mean(),
                "dist_std": dists.std(),

                "goal_rate": n_goals/n_episodes,
                "own_goal_rate": n_own_goals/n_episodes,
                "ball_out_rate": n_ball_out/n_episodes,
            }
        )


    # --- plots --- #
    model_names = [s["name"] for s in summaries]
    metrics = [
        ("reward_mean", "Mean Reward"),
        ("goal_rate", "Goal Rate"),
        ("own_goal_rate", "Own Goal Rate"),
        ("ball_out_rate", "Ball Out Rate"),
        ("dist_mean", "Final Distance To Goal (m)"),
        ("steps_mean", "Episode Length (steps)"),
    ]

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(model_names)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):

        values = [s[key] for s in summaries]

        ax.bar(
            model_names,
            values,
            color=colors,
            edgecolor="black"
        )

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # rotate labels if long
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Model Comparison", fontsize=16)
    plt.tight_layout()

    plt.show()



def compare_current_vs_backup(robot: str = "viper") -> None:
    """Evaluate current final_model vs backup_1v0 model on the same fixed phases.

    Prints a side-by-side goal-rate table so you can pick the stronger model
    for the article/video.  Runs in a single Webots session.
    """
    current = os.path.join(_CKPT_DIR, f"final_model_{robot}")
    backup  = os.path.join(_CKPT_DIR, "backup_1v0", f"final_model_{robot}")
    models  = [("current (warm-start)", current)]
    if os.path.isfile(backup + ".zip"):
        models.append(("backup (pre-warm-start)", backup))
    else:
        print(f"[compare] backup not found at {backup}.zip — só avalio o atual.")

    table = {}   # (label, phase) -> goal_rate
    for label, path in models:
        for ph in COMPARE_PHASES:
            print(f"\n{'='*60}\n {label}  |  PHASE {ph}\n{'='*60}")
            res = model_evaluate(path, COMPARE_EPISODES, DETERMINISTIC,
                                 eval_phase=ph)
            s = _summarise(res, COMPARE_EPISODES, label=f"{label} · phase {ph}")
            table[(label, ph)] = s["goal_rate"]

    print(f"\n{'='*60}\n COMPARISON — goal rate by phase ({robot})\n{'='*60}")
    header = "model".ljust(26) + "".join(f"  ph{p:>2}" for p in COMPARE_PHASES)
    print(header)
    for label, _ in models:
        row = label.ljust(26) + "".join(
            f"  {table[(label, p)]*100:>3.0f}%" for p in COMPARE_PHASES)
        print(row)


def run_eval(model_path: str = MODEL_PATH) -> None:
    """Top-level eval used by the Webots supervisor (MODE='eval').

    EVAL_PHASE is None  → sweep all phases 0..7 (honest, full picture).
    EVAL_PHASE is 0..7  → single fixed phase with N_EPISODES episodes.
    Then, if RUN_RANDOM_BASELINE, evaluates a random agent on phase 7 for
    the baseline comparison required by the objectives document.
    """
    if COMPARE_WITH_BACKUP:
        compare_current_vs_backup("viper")
        return
    if EVAL_PHASE is None:
        evaluate_phase_sweep(model_path, SWEEP_EPISODES_PER_PHASE, DETERMINISTIC)
    else:
        model_evaluate(model_path, N_EPISODES, DETERMINISTIC,
                       eval_phase=EVAL_PHASE)

    if RUN_RANDOM_BASELINE:
        print(f"\n{'='*60}\n RANDOM-AGENT BASELINE (phase 7)\n{'='*60}")
        res = model_evaluate(model_path, max(10, SWEEP_EPISODES_PER_PHASE),
                             DETERMINISTIC, eval_phase=7, random_agent=True)
        _summarise(res, max(10, SWEEP_EPISODES_PER_PHASE),
                   label="Random baseline (phase 7)")


# --- CLI --- #
def _cli() -> None:
    """Parse CLI args and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SoccerEnv model.")
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to .zip model file (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of evaluation episodes (default: {N_EPISODES})"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions (default: deterministic)"
    )
    args = parser.parse_args()
    model_evaluate(args.model, args.episodes, not args.stochastic)





# ------------ #
# --- MAIN --- #
# ------------ #

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        _cli()
    else:
        run_eval(MODEL_PATH)
