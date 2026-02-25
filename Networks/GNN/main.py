import time
import torch
from typing import Optional

from args import Args
from logger import Logger
from collections import deque
from network import GR_QNetwork
from minigrid.world import Agent
from minigrid.environment import MultiGridEnv
from utils import print_info


def train(
    env: MultiGridEnv, agents: list[Agent], network: GR_QNetwork, logger: Optional[Logger], args: Args
) -> torch.Tensor:
    training_start_time = time.time()

    progress = 0
    step_count = 0
    episode_count = 0
    eps = args.eps_start
    batch_size = len(agents)
    losses = deque(maxlen=100)
    total_scores = deque(maxlen=100)
    steps_per_episode = deque(maxlen=100)
    seen_percentages = deque(maxlen=100)
    agent_ids = torch.tensor([agent.id for agent in agents], device=args.device)

    while step_count < args.total_steps:
        episode_start_time = time.time()
        episode_count += 1
        scores = torch.zeros(len(agents), device=args.device)

        # Get initial observations in batched form
        obs_batch, node_obs_batch, adj_batch = env.reset(args.render)

        # Update network bias (PER)
        if step_count % 100 == 0:
            network.update_beta(step_count, args.total_steps, args.prio_b)

        for step in range(args.episode_steps):
            step_count += 1

            # Get actions for all agents in one forward pass
            actions = network.action(obs_batch, node_obs_batch, adj_batch, agent_ids, eps, args.debug)

            # Step environment with batched actions
            next_obs_batch, next_node_obs_batch, next_adj_batch, rewards, dones, info = env.step(actions, args.render)

            if args.debug:
                for i in range(batch_size):
                    print_info(
                        agent_ids[i],
                        obs_batch[i],
                        node_obs_batch[i],
                        adj_batch[i],
                        actions[i],
                        rewards[i],
                        next_obs_batch[i],
                        next_node_obs_batch[i],
                        next_adj_batch[i],
                        dones[i],
                    )

            # Process all experiences in one batch
            loss = network.step(
                agent_ids,
                obs_batch,
                node_obs_batch,
                adj_batch,
                actions,
                rewards,
                next_obs_batch,
                next_node_obs_batch,
                next_adj_batch,
                dones,
            )

            if loss is not None and step_count > 10000:
                losses.append(loss.cpu().item())
                torch.cuda.empty_cache()

            # Update observations
            scores += rewards
            obs_batch = next_obs_batch
            node_obs_batch = next_node_obs_batch
            adj_batch = next_adj_batch

            # Update epsilon
            if eps > args.eps_end:
                eps -= args.eps_decay
            else:
                eps = args.eps_end

            if any(dones):
                seen_percentages.append(info["seen_percentage"])
                break

        episode_time = time.time() - episode_start_time
        total_training_time = time.time() - training_start_time

        # Properly handle tensor operations
        batch_scores = scores.sum()
        total_scores.append(batch_scores.item())
        steps_per_episode.append(step)

        # Calculate averages
        if logger:
            avg_rewards = torch.tensor(list(total_scores), device=args.device, dtype=torch.float32).mean().item()
            avg_steps = torch.tensor(list(steps_per_episode), device=args.device, dtype=torch.float32).mean().item()
            avg_loss = (
                torch.tensor(list(losses), device=args.device, dtype=torch.float32).mean().item() if losses else 0.0
            )
            avg_seen = torch.tensor(list(seen_percentages), device=args.device, dtype=torch.float32).mean().item()
            steps_per_second = step_count / total_training_time

            logger.log_metrics(
                total_steps=step_count,
                episodes=episode_count,
                epsilon=eps,
                average_rewards=avg_rewards,
                average_steps_per_episode=avg_steps,
                average_loss=avg_loss,
                goals_collected=info.get("goals_collected", 0),
                goals_percentage=info.get("goals_percentage", 0),
                seen_percentage=avg_seen,
                episode_duration=episode_time,
                episode_duration_minutes=episode_time / 60,
                total_training_time=total_training_time,
                total_training_time_minutes=total_training_time / 60,
                total_training_time_hours=total_training_time / 3600,
                steps_per_second=steps_per_second,
            )
        else:
            # Calculate only necessary metrics for progress display
            avg_rewards = torch.tensor(list(total_scores), device=args.device, dtype=torch.float32).mean().item()
            avg_seen = torch.tensor(list(seen_percentages), device=args.device, dtype=torch.float32).mean().item()

        # Progress tracking and model saving
        current_progress = round((step_count / args.total_steps) * 100)

        if current_progress != progress:
            print(
                f"\r{current_progress}% | Eps: {eps:.2f} \tAvg Score: {avg_rewards:.2f} "
                f"\tAvg Seen: {avg_seen:.2f}% | Time: {total_training_time / 3600:.2f}h"
            )
            progress = current_progress
            torch.save(network.qnetwork_local.state_dict(), f"{args.title}.pt")

    # Save the learned policy
    torch.save(network.qnetwork_local.state_dict(), f"{args.title}.pt")
    return scores


def main() -> None:
    # Settings & Controls
    args = Args()
    agents = [Agent(id=i, args=args) for i in range(args.num_agents)]
    env = MultiGridEnv(args, agents)
    network = GR_QNetwork(args)

    if args.logger:
        logger = Logger(args)
    else:
        logger = None

    # Start Run
    if not args.load_policy:
        _ = train(env, agents, network, logger, args)

    # Close Environment
    if args.logger:
        logger.close()
    env.close()


if __name__ == "__main__":
    main()
