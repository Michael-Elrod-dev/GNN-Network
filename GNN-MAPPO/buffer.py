"""
GraphReplayBuffer: on-policy rollout buffer for GNN-MAPPO.
Adapted from InforMARL/onpolicy/utils/graph_buffer.py.

Key changes vs. original:
  - gym.Space args replaced with explicit shape/dim ints.
  - np.int replaced with np.int32 (deprecated in NumPy 1.24).
  - Only feed_forward_generator kept (recurrent generators removed).
  - available_actions sized for Discrete(4).
"""

import numpy as np
import torch
from typing import Optional, Tuple, Generator


def _flat_copy(arr: np.ndarray) -> np.ndarray:
    """Return a writeable contiguous copy shaped for PPO mini-batching."""
    return arr.reshape(-1, *arr.shape[3:]).copy()


class GraphReplayBuffer:
    """
    On-policy rollout buffer for one epoch of experience.

    Shapes (T, R, A, ...):
        T = episode_length      (175)
        R = n_rollout_threads   (8)
        A = num_agents          (8)

    Buffer arrays have T+1 slots (first slot = bootstrap / current obs).

    Args:
        args:               hyperparameter namespace
        num_agents:         number of agents (8)
        obs_dim:            local obs dim (8)
        cent_obs_dim:       centralized obs dim (64)
        node_obs_shape:     (num_entities, node_feat_dim) = (51, 9)
        adj_shape:          (num_entities, num_entities) = (51, 51)
        agent_id_dim:       1
        share_agent_id_dim: num_agents = 8
        action_dim:         1  (stored action index)
    """

    def __init__(
        self,
        args,
        num_agents: int,
        obs_dim: int,
        cent_obs_dim: int,
        node_obs_shape: Tuple[int, int],
        adj_shape: Tuple[int, int],
        agent_id_dim: int,
        share_agent_id_dim: int,
        action_dim: int,
    ):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        T = self.episode_length
        R = self.n_rollout_threads
        A = num_agents

        # Observations
        self.obs = np.zeros((T + 1, R, A, obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((T + 1, R, A, cent_obs_dim), dtype=np.float32)
        self.node_obs = np.zeros((T + 1, R, A, *node_obs_shape), dtype=np.float32)
        self.adj = np.zeros((T + 1, R, A, *adj_shape), dtype=np.float32)
        self.agent_id = np.zeros((T + 1, R, A, agent_id_dim), dtype=np.int32)
        self.share_agent_id = np.zeros((T + 1, R, A, share_agent_id_dim), dtype=np.int32)

        # Actions — Discrete(4): 4 available actions per step
        self.available_actions = np.ones((T + 1, R, A, 4), dtype=np.float32)

        # Actions + log probs
        self.actions = np.zeros((T, R, A, action_dim), dtype=np.float32)
        self.action_log_probs = np.zeros((T, R, A, action_dim), dtype=np.float32)

        # Rewards / masks / active
        self.rewards = np.zeros((T, R, A, 1), dtype=np.float32)
        self.masks = np.ones((T + 1, R, A, 1), dtype=np.float32)
        self.bad_masks = np.ones((T + 1, R, A, 1), dtype=np.float32)
        self.active_masks = np.ones((T + 1, R, A, 1), dtype=np.float32)

        # Critic predictions
        self.value_preds = np.zeros((T + 1, R, A, 1), dtype=np.float32)
        self.returns = np.zeros((T + 1, R, A, 1), dtype=np.float32)

        # RNN states (allocated even when unused; shape R×A×recurrent_N×hidden)
        self.rnn_states = np.zeros((T + 1, R, A, args.recurrent_N, args.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.step = 0

    # ------------------------------------------------------------------
    # Insert one environment step
    # ------------------------------------------------------------------

    def insert(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        node_obs: np.ndarray,
        adj: np.ndarray,
        agent_id: np.ndarray,
        share_agent_id: np.ndarray,
        rnn_states: np.ndarray,
        rnn_states_critic: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        value_preds: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        bad_masks: Optional[np.ndarray] = None,
        active_masks: Optional[np.ndarray] = None,
        available_actions: Optional[np.ndarray] = None,
    ):
        t = self.step + 1
        self.obs[t] = obs.copy()
        self.share_obs[t] = share_obs.copy()
        self.node_obs[t] = node_obs.copy()
        self.adj[t] = adj.copy()
        self.agent_id[t] = agent_id.copy()
        self.share_agent_id[t] = share_agent_id.copy()
        self.rnn_states[t] = rnn_states.copy()
        self.rnn_states_critic[t] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[t] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[t] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[t] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[t] = available_actions.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last step to slot 0 to prepare for next rollout."""
        self.obs[0] = self.obs[-1].copy()
        self.share_obs[0] = self.share_obs[-1].copy()
        self.node_obs[0] = self.node_obs[-1].copy()
        self.adj[0] = self.adj[-1].copy()
        self.agent_id[0] = self.agent_id[-1].copy()
        self.share_agent_id[0] = self.share_agent_id[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

    # ------------------------------------------------------------------
    # GAE return computation
    # ------------------------------------------------------------------

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns (with GAE if enabled).
        next_value: (R, A, 1) numpy array from critic bootstrap.
        """
        self.value_preds[-1] = next_value

        if self._use_gae:
            gae = 0.0
            for step in reversed(range(self.episode_length)):
                if value_normalizer is not None:
                    next_val = value_normalizer.denormalize(self.value_preds[step + 1])
                    cur_val = value_normalizer.denormalize(self.value_preds[step])
                else:
                    next_val = self.value_preds[step + 1]
                    cur_val = self.value_preds[step]

                if self._use_proper_time_limits:
                    delta = self.rewards[step] + self.gamma * next_val * self.bad_masks[step + 1] - cur_val
                    gae = delta + self.gamma * self.gae_lambda * self.bad_masks[step + 1] * gae
                    self.returns[step] = gae + cur_val
                else:
                    delta = self.rewards[step] + self.gamma * next_val * self.masks[step + 1] - cur_val
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + cur_val
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.episode_length)):
                if self._use_proper_time_limits:
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.bad_masks[step + 1] + self.rewards[step]
                    )
                else:
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    # ------------------------------------------------------------------
    # Mini-batch generator (feed-forward only)
    # ------------------------------------------------------------------

    def feed_forward_generator(
        self,
        advantages: torch.Tensor,
        num_mini_batch: int,
    ) -> Generator:
        """
        Yield mini-batches of (T * R * A) samples.
        advantages: (T, R, A, 1) torch tensor (pre-computed before PPO epochs).
        """
        T = self.episode_length
        R = self.n_rollout_threads
        A = self.agent_id.shape[2]
        batch_size = T * R * A
        assert batch_size >= num_mini_batch, f"batch_size ({batch_size}) < num_mini_batch ({num_mini_batch})"
        mini_batch_size = batch_size // num_mini_batch

        # Flatten (T, R, A, ...) → (T*R*A, ...)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        node_obs = self.node_obs[:-1].reshape(-1, *self.node_obs.shape[3:])
        adj = self.adj[:-1].reshape(-1, *self.adj.shape[3:])
        agent_id = self.agent_id[:-1].reshape(-1, *self.agent_id.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[3:])
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[3:])
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[3:])
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[3:])
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[3:])
        available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[3:])
        # trainer re-computes advantages from return_batch - value_preds_batch for normalisation
        # but we still pass return_batch so the trainer can do its own norm.

        rand_idx = np.random.permutation(batch_size)
        sampler = [rand_idx[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            yield (
                share_obs[indices],
                obs[indices],
                node_obs[indices],
                adj[indices],
                agent_id[indices],
                rnn_states[indices],
                rnn_states_critic[indices],
                actions[indices],
                value_preds[indices],
                returns[indices],
                masks[indices],
                active_masks[indices],
                action_log_probs[indices],
                available_actions[indices],
            )
