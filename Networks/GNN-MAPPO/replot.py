import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


DQN_STEPS = [0, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
DQN_PCT = [0, 6, 11, 21, 28, 35, 42, 42, 42, 42, 42]

GNN_STEPS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]
GNN_PCT = [0, 17, 35, 46, 53, 58, 70, 77, 78, 82, 86, 86, 88, 89, 89]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="comparison_data.csv",
        help="CSV file written by plot_comparison.py (default: comparison_data.csv)",
    )
    parser.add_argument("--output", type=str, default="replot.png", help="Output filename (default: replot.png)")
    parser.add_argument(
        "--include-gnn-mappo",
        action="store_true",
        default=False,
        help="Include the GNN-MAPPO data series in the plot (requires --data CSV file)",
    )
    return parser.parse_args()


def main():
    cli = parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))

    episode_length = 200  # default x-axis limit when GNN-MAPPO is not loaded

    if cli.include_gnn_mappo:
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
        print(f"Final GNN-MAPPO goal %: {mappo_plot[-1]:.1f}%")

        ax.plot(steps, mappo_plot, color="#2ca02c", linestyle="-", linewidth=2, label="GNN-MAPPO")

    ax.plot(DQN_STEPS, DQN_PCT, color="#1f77b4", linestyle="-", linewidth=1.5, label="DQN")
    ax.plot(GNN_STEPS, GNN_PCT, color="#ff7f0e", linestyle="-", linewidth=2, label="GNN")

    ax.set_title("8 Agents | 45 Objectives | 30×30 Grid")
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
