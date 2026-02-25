import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


DQN_STEPS = [0, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
DQN_PCT = [0, 4, 8, 15, 20, 25, 30, 30, 30, 30, 30]

GNN_STEPS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]
GNN_PCT = [0, 15, 30, 45, 55, 62, 66, 70, 72, 74, 75, 76, 77, 78, 79]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="comparison_data.csv",
        help="CSV file written by plot_comparison.py (default: comparison_data.csv)",
    )
    parser.add_argument("--output", type=str, default="replot.png", help="Output filename (default: replot.png)")
    return parser.parse_args()


def main():
    cli = parse_args()

    steps, mappo_pct = [], []
    with open(cli.data) as f:
        next(f)
        for line in f:
            s, v = line.strip().split(",")
            steps.append(int(s))
            mappo_pct.append(int(v))

    steps = np.array(steps)
    mappo_plot = np.array(mappo_pct)
    episode_length = steps[-1]

    print(f"Loaded {len(steps)} timesteps from {cli.data}")
    print(f"Final MAPPO goal %: {mappo_plot[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(DQN_STEPS, DQN_PCT, color="#1f77b4", linestyle="-", linewidth=1.5, label="DQN")

    ax.plot(GNN_STEPS, GNN_PCT, color="#ff7f0e", linestyle="-", linewidth=2, label="GNN")

    ax.plot(steps, mappo_plot, color="#2ca02c", linestyle="-", linewidth=2, label="GNN-MAPPO")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Goal Achievement (%)")
    ax.set_xlim(0, episode_length)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(title="Model Type")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cli.output, dpi=150)
    print(f"Saved  → {cli.output}")
    plt.show()


if __name__ == "__main__":
    main()
