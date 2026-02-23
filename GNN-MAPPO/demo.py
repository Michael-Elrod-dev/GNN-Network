"""
Visual demo of a trained GNN-MAPPO policy.

Loads the latest saved actor checkpoint and runs it on a single environment
with a pygame render window. Policy runs deterministically (no exploration).

Usage:
    python demo.py                          # latest checkpoint
    python demo.py --episode 500           # specific checkpoint episode
    python demo.py --delay 0.05            # faster (default 0.1s per step)
    python demo.py --episodes 5            # run N episodes then exit
"""

import sys
import time
import argparse
import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pygame
from args import Args
from actor_critic import GR_Actor
from minigrid.environment import MultiGridEnv
from minigrid.world import Agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=None, help="Checkpoint episode to load (default: latest)")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between steps (default: 0.1)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run (default: 10)")
    parser.add_argument(
        "--deterministic", type=lambda x: x.lower() != "false", default=False, help="Greedy policy (default: False)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Softmax temperature; <1 sharpens, >1 smooths (default: 1.0)"
    )
    return parser.parse_args()


def load_actor(args, device, episode=None):
    """Load the actor from checkpoint."""
    checkpoint_dir = Path("checkpoints") / args.title
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints found at {checkpoint_dir}")

    if episode is not None:
        path = checkpoint_dir / f"actor_ep{episode}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint for episode {episode}")
    else:
        # Find latest
        files = sorted(checkpoint_dir.glob("actor_ep*.pt"), key=lambda p: int(p.stem.split("ep")[1]))
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


def draw_stats(window, font, step, goals_collected, num_goals, episode, total_episodes):
    """Draw a semi-transparent stats bar at the bottom of the window."""
    W, H = window.get_size()
    bar_h = 36
    bar = pygame.Surface((W, bar_h), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 160))
    window.blit(bar, (0, H - bar_h))

    text = (
        f"Episode {episode}/{total_episodes}  |  "
        f"Step {step:3d}/175  |  "
        f"Goals {goals_collected}/{num_goals}  ({goals_collected / num_goals * 100:.0f}%)"
    )
    surf = font.render(text, True, (255, 255, 255))
    window.blit(surf, (10, H - bar_h + 8))
    pygame.display.flip()


def main():
    cli = parse_args()
    args = Args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    print(f"Device: {args.device}")
    actor = load_actor(args, device, episode=cli.episode)

    # Build a single env on the main thread (required for pygame)
    agents = [Agent(id=i, args=args) for i in range(args.num_agents)]
    env = MultiGridEnv(args, agents)

    # Fixed agent_id and rnn/mask placeholders
    A = args.num_agents
    ag_id = np.arange(A, dtype=np.int32).reshape(A, 1)
    rnn_states = np.zeros((A, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((A, 1), dtype=np.float32)

    # Init pygame font (after first render_env creates the window)
    pygame.init()
    font = pygame.font.SysFont("monospace", 15, bold=True)

    results = []

    for ep in range(cli.episodes):
        obs, node_obs, adj = env.reset()
        # obs:      (A, 8)
        # node_obs: (A, 51, 9)
        # adj:      (A, 51, 51)

        ep_goals = 0
        ep_steps = 0
        env.render_env()

        for step in range(args.episode_length):
            # Check for pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    pygame.quit()
                    sys.exit()

            # Get actions from policy
            with torch.no_grad():
                actions, _, _ = actor(
                    obs,
                    node_obs,
                    adj,
                    ag_id,
                    rnn_states,
                    masks,
                    deterministic=cli.deterministic,
                    temperature=cli.temperature,
                )
            actions_np = actions.cpu().numpy().flatten().astype(np.int32)  # (A,)

            obs, node_obs, adj, rewards, dones, info = env.step(actions_np)

            if info:
                ep_goals = info["goals_collected"]
            ep_steps = step + 1

            env.render_env()

            # Draw stats overlay
            if env.window is not None:
                draw_stats(
                    env.window,
                    font,
                    step=ep_steps,
                    goals_collected=env.num_collected,
                    num_goals=args.num_goals,
                    episode=ep + 1,
                    total_episodes=cli.episodes,
                )

            time.sleep(cli.delay)

            if dones.all():
                break

        pct = ep_goals / args.num_goals * 100
        results.append(pct)
        print(f"Episode {ep + 1:3d}/{cli.episodes}  Goals={ep_goals}/{args.num_goals}  ({pct:.1f}%)  Steps={ep_steps}")
        time.sleep(0.5)

    print(f"\nAverage over {cli.episodes} episodes: {np.mean(results):.1f}%")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
