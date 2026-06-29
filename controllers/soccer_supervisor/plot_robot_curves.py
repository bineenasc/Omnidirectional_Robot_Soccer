"""
plot_robot_curves.py
-------------------------------------------------------------------------------
Gera a curva POR ÉPOCA (recompensa média + taxa de gols) de cada robô a partir
de logs/episode_log.csv — no mesmo estilo das figuras geradas pelo train.py,
mas SEM precisar terminar o treino nem abrir o Webots.

Útil quando o run foi interrompido (o train.py só salva training_curves_<robot>.png
no fim de tudo). Lê o CSV de forma robusta (ignora bytes NUL de escrita parcial).

    py -3.13 plot_robot_curves.py            # gera p/ todos os robôs no CSV
    py -3.13 plot_robot_curves.py --robot viper
    py -3.13 plot_robot_curves.py --no-show

Saída: plots/training_curves_<robot>.png
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, "logs", "episode_log.csv")
_PLOT_DIR = os.path.join(_HERE, "plots")

COLORS = {"viper": "#2196F3", "titan": "#F44336"}


def _read_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise SystemExit(f"CSV nao encontrado: {path}\nRode um treino primeiro.")
    # Lê tolerando bytes NUL (CSV pode estar sendo escrito durante o treino).
    text = open(path, "r", errors="replace").read().replace("\x00", "")
    return list(csv.DictReader(text.splitlines()))


def _plot_robot(rows: list[dict], robot: str, show: bool) -> None:
    # Agrupa por "stage" (= índice de época no CSV) só deste robô.
    rewards: dict[int, list[float]] = defaultdict(list)
    goals: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("robot") != robot:
            continue
        try:
            st = int(float(r["stage"]))
            rewards[st].append(float(r["reward"]))
            goals[st].append(int(float(r["goal"])))
        except (KeyError, ValueError):
            continue

    if not rewards:
        print(f"[{robot}] sem episodios no CSV — pulado.")
        return

    stages = sorted(rewards.keys())
    # Reindexa para 1..N (época) para o eixo X ficar limpo.
    epochs = list(range(1, len(stages) + 1))
    mean_r = [sum(rewards[s]) / len(rewards[s]) for s in stages]
    goal_r = [100.0 * sum(goals[s]) / len(goals[s]) for s in stages]
    color = COLORS.get(robot, "#4CAF50")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Training curves — {robot.capitalize()}", fontweight="bold")

    ax1.bar(epochs, mean_r, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Mean episode reward")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(epochs, goal_r, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Goal rate (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(_PLOT_DIR, exist_ok=True)
    out = os.path.join(_PLOT_DIR, f"training_curves_{robot}.png")
    fig.savefig(out, dpi=150)
    print(f"[{robot}] {len(stages)} epocas | "
          f"recompensa media {sum(mean_r)/len(mean_r):.1f} | "
          f"gol medio {sum(goal_r)/len(goal_r):.1f}% -> {out}")
    if not show:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Curvas por epoca, por robo, a partir do CSV.")
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--robot", default=None, help="viper | titan (default: todos)")
    ap.add_argument("--no-show", action="store_true", help="so salva, nao abre janela")
    args = ap.parse_args()

    rows = _read_rows(args.csv)
    robots = [args.robot] if args.robot else ["viper", "titan"]
    for rb in robots:
        _plot_robot(rows, rb, show=not args.no_show)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
