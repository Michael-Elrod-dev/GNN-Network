"""
WandB logger for GNN-MAPPO.
Logs PPO training metrics + environment metrics.
"""

import wandb


class Logger:
    def __init__(self, args, project: str = "GNN-MAPPO"):
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            wandb.init(
                project=project,
                name=args.title,
                config={
                    # Environment
                    "num_agents": args.num_agents,
                    "num_goals": args.num_goals,
                    "num_obstacles": args.num_obstacles,
                    "grid_size": args.grid_size,
                    "agent_view_size": args.agent_view_size,
                    "max_edge_dist": args.max_edge_dist,
                    # PPO
                    "n_rollout_threads": args.n_rollout_threads,
                    "episode_length": args.episode_length,
                    "num_env_steps": args.num_env_steps,
                    "ppo_epoch": args.ppo_epoch,
                    "num_mini_batch": args.num_mini_batch,
                    "clip_param": args.clip_param,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "lr": args.lr,
                    "critic_lr": args.critic_lr,
                    "entropy_coef": args.entropy_coef,
                    "value_loss_coef": args.value_loss_coef,
                    # GNN
                    "gnn_hidden_size": args.gnn_hidden_size,
                    "gnn_num_heads": args.gnn_num_heads,
                    "gnn_layer_N": args.gnn_layer_N,
                    "node_obs_shape": args.node_obs_shape,
                    "actor_graph_aggr": args.actor_graph_aggr,
                    "critic_graph_aggr": args.critic_graph_aggr,
                    # MLP
                    "hidden_size": args.hidden_size,
                    "layer_N": args.layer_N,
                },
            )

    def log_metrics(
        self,
        episode: int,
        # PPO losses
        value_loss: float,
        policy_loss: float,
        dist_entropy: float,
        actor_grad_norm: float,
        critic_grad_norm: float,
        # Env metrics
        goals_collected: float,
        goals_percentage: float,
        seen_percentage: float,
        fps: float,
    ):
        if not self.use_wandb:
            return
        wandb.log(
            {
                "episode": episode,
                # PPO
                "Value Loss": value_loss,
                "Policy Loss": policy_loss,
                "Entropy": dist_entropy,
                "Actor Grad Norm": actor_grad_norm,
                "Critic Grad Norm": critic_grad_norm,
                # Env
                "Goals Collected": goals_collected,
                "Goals %": goals_percentage,
                "Grid Seen %": seen_percentage,
                "FPS": fps,
            },
            step=episode,
        )

    def finish(self):
        if self.use_wandb:
            wandb.finish()
