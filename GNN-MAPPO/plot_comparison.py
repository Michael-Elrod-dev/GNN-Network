"""
plot_comparison.py  –  Overlays live MAPPO results on hard-coded GNN / DQN curves.

Hard-coded values are read directly from the original Screenshot.png.

Usage:
    python plot_comparison.py                   # 10 eval episodes, latest checkpoint
    python plot_comparison.py --episodes 20     # smoother average
    python plot_comparison.py --episode 500     # specific checkpoint
    python plot_comparison.py --output out.png  # custom save path
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from args import Args
from actor_critic import GR_Actor
from minigrid.environment import MultiGridEnv
from minigrid.world import Agent


# ---------------------------------------------------------------------------
# Hard-coded reference curves (digitised from Screenshot.png)
# ---------------------------------------------------------------------------

# DQN: staircase rise, flat at ~30% from step 30 onward
DQN_STEPS = [0, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
DQN_PCT = [0, 4, 8, 15, 20, 25, 30, 30, 30, 30, 30]

# GNN: rapid sigmoid rise, asymptotes ~79%
GNN_STEPS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]
GNN_PCT = [0, 15, 30, 45, 55, 62, 66, 70, 72, 74, 75, 76, 77, 78, 79]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=None, help="Checkpoint episode to load (default: latest)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes to average (default: 10)")
    parser.add_argument(
        "--output", type=str, default="comparison.png", help="Output filename (default: comparison.png)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Softmax temperature for action sampling (default: 0.5)"
    )
    return parser.parse_args()


def load_actor(args, device, episode=None):
    checkpoint_dir = Path("checkpoints") / args.title
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints found at {checkpoint_dir}")

    if episode is not None:
        path = checkpoint_dir / f"actor_ep{episode}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint for episode {episode}")
    else:
        files = sorted(
            checkpoint_dir.glob("actor_ep*.pt"),
            key=lambda p: int(p.stem.split("ep")[1]),
        )
        if not files:
            raise FileNotFoundError(f"No actor checkpoints in {checkpoint_dir}")
        path = files[-1]

    print(f"Loading: {path}")
    actor = GR_Actor(
        args,
        obs_dim=8,
        node_obs_dim=args.node_obs_shape,
        edge_dim=args.edge_dim,
        action_dim=4,
        device=device,
        split_batch=False,
    )
    actor.load_state_dict(torch.load(path, map_location=device))
    actor.eval()
    return actor


def run_episode(actor, env, args, device, temperature=0.5):
    """
    Run one headless episode and return a per-timestep goal-achievement array.
    Uses info["goals_collected"] at episode end to avoid reading env.num_collected
    after the internal auto-reset zeroes it.
    """
    A = args.num_agents
    ag_id = np.arange(A, dtype=np.int32).reshape(A, 1)
    rnn_states = np.zeros((A, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((A, 1), dtype=np.float32)

    obs, node_obs, adj = env.reset()
    curve = []

    for _ in range(args.episode_length):
        with torch.no_grad():
            actions, _, _ = actor(
                obs,
                node_obs,
                adj,
                ag_id,
                rnn_states,
                masks,
                deterministic=False,
                temperature=temperature,
            )
        actions_np = actions.cpu().numpy().flatten().astype(np.int32)
        obs, node_obs, adj, _rewards, dones, info = env.step(actions_np)

        # When episode ends, env auto-resets and zeros num_collected,
        # so read from info instead.
        if info:
            pct = info["goals_collected"] / args.num_goals * 100
        else:
            pct = env.num_collected / args.num_goals * 100
        curve.append(pct)

        if dones.all():
            # Pad remaining steps with the final value
            final = curve[-1]
            curve.extend([final] * (args.episode_length - len(curve)))
            break

    return np.array(curve[: args.episode_length])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cli = parse_args()
    args = Args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    actor = load_actor(args, device, cli.episode)

    agents = [Agent(id=i, args=args) for i in range(args.num_agents)]
    env = MultiGridEnv(args, agents)

    print(f"Running {cli.episodes} eval episodes (headless)...")
    curves = []
    for ep in range(cli.episodes):
        curve = run_episode(actor, env, args, device, temperature=cli.temperature)
        print(f"  Episode {ep + 1:3d}/{cli.episodes}  final={curve[-1]:.1f}%")
        curves.append(curve)

    env.close()

    steps = np.arange(1, args.episode_length + 1)

    mappo_plot = np.mean(curves, axis=0)
    mappo_label = "GNN-MAPPO"

    # --- Save MAPPO curve to text file ---
    csv_path = Path(cli.output).stem + "_data.csv"
    with open(csv_path, "w") as f:
        f.write("step,goal_pct\n")
        for s, v in zip(steps, mappo_plot):
            f.write(f"{s},{round(v)}\n")
    print(f"Data  → {csv_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(DQN_STEPS, DQN_PCT, color="#1f77b4", linestyle="-", linewidth=1.5, label="DQN")

    ax.plot(GNN_STEPS, GNN_PCT, color="#ff7f0e", linestyle="-", linewidth=2, label="GNN - Buffer")

    ax.plot(steps, mappo_plot, color="#2ca02c", linestyle="-", linewidth=2, label=mappo_label)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Goal Achievement (%)")
    ax.set_xlim(0, args.episode_length)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(title="Model Type")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cli.output, dpi=150)
    print(f"\nSaved → {cli.output}")
    plt.show()


if __name__ == "__main__":
    main()
