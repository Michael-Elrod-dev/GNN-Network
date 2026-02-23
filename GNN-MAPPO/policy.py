"""
GR_MAPPOPolicy: wraps GR_Actor + GR_Critic with Adam optimizers.
Adapted from InforMARL/onpolicy/algorithms/graph_MAPPOPolicy.py.

Key changes vs. original:
  - gym.Space args replaced with explicit int dims.
  - AMP / scaler removed.
"""

import torch
from torch.optim import Adam

from actor_critic import GR_Actor, GR_Critic
from utils import update_linear_schedule


class GR_MAPPOPolicy:
    """
    Policy class for GR-MAPPO.

    Args:
        args:          hyperparameter namespace
        obs_dim:       local observation dim (8)
        cent_obs_dim:  centralized obs dim (64)
        node_obs_dim:  per-entity node feature dim (9)
        edge_dim:      edge attribute dim (1)
        action_dim:    number of discrete actions (4)
        device:        torch device
    """

    def __init__(
        self,
        args,
        obs_dim: int,
        cent_obs_dim: int,
        node_obs_dim: int,
        edge_dim: int,
        action_dim: int,
        device=torch.device("cpu"),
    ):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_dim = obs_dim
        self.cent_obs_dim = cent_obs_dim
        self.node_obs_dim = node_obs_dim
        self.edge_dim = edge_dim
        self.action_dim = action_dim

        self.actor = GR_Actor(
            args,
            obs_dim=obs_dim,
            node_obs_dim=node_obs_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            device=device,
            split_batch=args.split_batch,
            max_batch_size=args.max_batch_size,
        )
        self.critic = GR_Critic(
            args,
            cent_obs_dim=cent_obs_dim,
            node_obs_dim=node_obs_dim,
            edge_dim=edge_dim,
            device=device,
            split_batch=args.split_batch,
            max_batch_size=args.max_batch_size,
        )

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode: int, episodes: int):
        """Linear lr decay."""
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        cent_obs,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        """Collect rollout: sample actions + values."""
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            temperature,
        )
        values, rnn_states_critic = self.critic(
            cent_obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_critic,
            masks,
        )
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(
        self,
        cent_obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_critic,
        masks,
    ):
        """Compute bootstrap value for last step."""
        values, _ = self.critic(
            cent_obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_critic,
            masks,
        )
        return values

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Evaluate actions for PPO update (returns log_probs, entropy, values)."""
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
        )
        values, _ = self.critic(
            cent_obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_critic,
            masks,
        )
        return values, action_log_probs, dist_entropy

    def act(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        """Deterministic or stochastic action (no critic)."""
        actions, _, rnn_states_actor = self.actor(
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            temperature,
        )
        return actions, rnn_states_actor

    def save(self, save_dir: str, episode: int):
        import os

        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{save_dir}/actor_ep{episode}.pt")
        torch.save(self.critic.state_dict(), f"{save_dir}/critic_ep{episode}.pt")

    def restore(self, model_dir: str):
        import glob

        actor_files = sorted(glob.glob(f"{model_dir}/actor_ep*.pt"))
        critic_files = sorted(glob.glob(f"{model_dir}/critic_ep*.pt"))
        if actor_files:
            self.actor.load_state_dict(torch.load(actor_files[-1], map_location=self.device))
        if critic_files:
            self.critic.load_state_dict(torch.load(critic_files[-1], map_location=self.device))
