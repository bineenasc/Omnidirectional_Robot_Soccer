"""
plot_results.py
-------------------------------------------------------------------------------
Curva de aprendizado POR EPISODIO (com media movel) a partir do CSV escrito por
train.py (logs/episode_log.csv). NAO precisa fechar o Webots - so le o arquivo.

    py -3.13 plot_results.py
    py -3.13 plot_results.py --window 100
    py -3.13 plot_results.py --no-show --out fig.png
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, "logs", "episode_log.csv")
_DEFAULT_OUT = os.path.join(_HERE, "plots", "learning_curve.png")


def _rolling(values, window):
    out, acc = [], []
    s = 0.0
    for v in values:
        acc.append(v)
        s += v
        if len(acc) > window:
            s -= acc.pop(0)
        out.append(s / len(acc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Curva de aprendizado por episodio.")
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--window", type=int, default=50, help="janela da media movel")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--no-show", action="store_true", help="so salva, nao abre janela")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV nao encontrado: {args.csv}\nRode um treino primeiro.")

    ts, rew, goal = [], [], []
    # Le tolerando bytes NUL (CSV pode estar sendo escrito durante o treino).
    _text = open(args.csv, "r", errors="replace").read().replace("\x00", "")
    for row in csv.DictReader(_text.splitlines()):
        try:
            ts.append(float(row["timestep"]))
            rew.append(float(row["reward"]))
            goal.append(int(float(row["goal"])))
        except (KeyError, ValueError):
            continue
    if not rew:
        raise SystemExit("CSV vazio - nenhum episodio ainda.")

    w = args.window
    ep = list(range(1, len(rew) + 1))
    rew_ma = _rolling(rew, w)
    goal_ma = [100.0 * g for g in _rolling(goal, w)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(ep, rew, alpha=0.20, color="tab:blue", label="recompensa bruta")
    ax.plot(ep, rew_ma, color="tab:blue", linewidth=2, label=f"media movel ({w})")
    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa acumulada")
    ax.set_title("Curva de aprendizado"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ts, rew_ma, color="tab:green", linewidth=2)
    ax.set_xlabel("Timesteps totais"); ax.set_ylabel(f"Recompensa (media movel {w})")
    ax.set_title("Eficiencia amostral"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep, goal_ma, color="tab:orange", linewidth=2)
    ax.set_xlabel("Episodio"); ax.set_ylabel("Taxa de gols (%)")
    ax.set_ylim(0, 100); ax.set_title("Taxa de gols (media movel)"); ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Grafico salvo em {args.out}")

    print("\n=== Resumo ===")
    print(f"Episodios:           {len(rew)}")
    print(f"Recompensa media:    {sum(rew)/len(rew):.2f}")
    print(f"Melhor episodio:     {max(rew):.2f}")
    tail = rew[-w:]
    print(f"Ultimos {len(tail)} ep.:     {sum(tail)/len(tail):.2f}")
    print(f"Taxa de gols global: {sum(goal)/len(goal):.1%}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
