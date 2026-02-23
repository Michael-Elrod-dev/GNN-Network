"""
Runner: training loop + rollout collection for GNN-MAPPO.
Flattened from InforMARL/onpolicy/runner/shared/graph_mpe_runner.py + base_runner.py.

Environment:
  - 31×31 grid, 8 agents, 43 goals, no obstacles
  - reset() → (obs, node_obs, adj)                              shapes: (A,8), (A,51,9), (A,51,51)
  - step()  → (obs, node_obs, adj, rewards, dones, info)

Vectorised env (GraphSubprocVecEnv) adds ag_id and stacks across R threads:
  reset() → (R,A,8), (R,A,1), (R,A,51,9), (R,A,51,51)
  step()  → (R,A,8), (R,A,1), (R,A,51,9), (R,A,51,51), (R,A,1), (R,A,1), infos
"""

import os
import time
import numpy as np
import torch
from typing import List, Dict, Any

from env_wrappers import GraphSubprocVecEnv, GraphDummyVecEnv
from policy import GR_MAPPOPolicy
from trainer import GR_MAPPO
from buffer import GraphReplayBuffer
from logger import Logger
from utils import _t2n


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(args):
    """Return a thunk that builds a single MultiGridEnv."""

    def _thunk():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from minigrid.environment import MultiGridEnv
        from minigrid.world import Agent

        agents = [Agent(id=i, args=args) for i in range(args.num_agents)]
        return MultiGridEnv(args, agents)

    return _thunk


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # --- Build vectorised envs ---
        env_fns = [make_env(args) for _ in range(args.n_rollout_threads)]
        if args.n_rollout_threads == 1:
            self.envs = GraphDummyVecEnv(env_fns)
        else:
            self.envs = GraphSubprocVecEnv(env_fns)

        # --- Dims ---
        self.num_agents = args.num_agents
        self.obs_dim = 8  # local obs features
        self.cent_obs_dim = self.obs_dim * self.num_agents  # 64
        self.node_obs_dim = args.node_obs_shape  # 9
        self.edge_dim = args.edge_dim  # 1
        self.action_dim = 4  # Discrete(4)
        self.agent_id_dim = 1
        self.share_agent_id_dim = self.num_agents  # 8

        # node_obs_shape (tuple) for buffer
        num_entities = args.num_agents + args.num_goals  # 51
        self.node_obs_shape = (num_entities, self.node_obs_dim)  # (51, 9)
        self.adj_shape = (num_entities, num_entities)  # (51, 51)

        # --- Policy / Trainer ---
        self.policy = GR_MAPPOPolicy(
            args,
            obs_dim=self.obs_dim,
            cent_obs_dim=self.cent_obs_dim,
            node_obs_dim=self.node_obs_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        if args.model_dir is not None:
            self.policy.restore(args.model_dir)

        self.trainer = GR_MAPPO(args, self.policy, device=self.device)

        # --- Buffer ---
        self.buffer = GraphReplayBuffer(
            args,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            cent_obs_dim=self.cent_obs_dim,
            node_obs_shape=self.node_obs_shape,
            adj_shape=self.adj_shape,
            agent_id_dim=self.agent_id_dim,
            share_agent_id_dim=self.share_agent_id_dim,
            action_dim=1,
        )

        # --- Logger ---
        self.logger = Logger(args)

        # Derived counts
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.num_env_steps = args.num_env_steps
        self.save_interval = args.save_interval
        self.log_interval = args.log_interval

        self.episodes = int(self.num_env_steps // self.episode_length // self.n_rollout_threads)

    # ------------------------------------------------------------------
    # Warmup: populate buffer slot 0 from env.reset()
    # ------------------------------------------------------------------

    def warmup(self):
        print("Starting envs (spawning subprocesses)...", flush=True)
        obs, ag_id, node_obs, adj = self.envs.reset()
        print("Envs ready.", flush=True)
        # obs:      (R, A, 8)
        # ag_id:    (R, A, 1)
        # node_obs: (R, A, 51, 9)
        # adj:      (R, A, 51, 51)

        share_obs = self._make_share_obs(obs)  # (R, A, 64)
        share_agent_id = self._make_share_agent_id(ag_id)  # (R, A, 8)

        self.buffer.obs[0] = obs
        self.buffer.share_obs[0] = share_obs
        self.buffer.node_obs[0] = node_obs
        self.buffer.adj[0] = adj
        self.buffer.agent_id[0] = ag_id
        self.buffer.share_agent_id[0] = share_agent_id

    # ------------------------------------------------------------------
    # Collect one rollout step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(self, step: int):
        R = self.n_rollout_threads
        A = self.num_agents

        self.trainer.prep_rollout()

        # Flatten (R, A, ...) → (R*A, ...)
        cent_obs = self.buffer.share_obs[step].reshape(R * A, -1)
        obs = self.buffer.obs[step].reshape(R * A, -1)
        node_obs = self.buffer.node_obs[step].reshape(R * A, *self.node_obs_shape)
        adj = self.buffer.adj[step].reshape(R * A, *self.adj_shape)
        agent_id = self.buffer.agent_id[step].reshape(R * A, -1)
        rnn_states = self.buffer.rnn_states[step].reshape(R * A, self.args.recurrent_N, self.args.hidden_size)
        rnn_states_critic = self.buffer.rnn_states_critic[step].reshape(
            R * A, self.args.recurrent_N, self.args.hidden_size
        )
        masks = self.buffer.masks[step].reshape(R * A, -1)

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.policy.get_actions(
            cent_obs,
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states,
            rnn_states_critic,
            masks,
        )

        # Reshape back to (R, A, ...)
        values = _t2n(values).reshape(R, A, 1)
        actions = _t2n(actions).reshape(R, A, 1)
        action_log_probs = _t2n(action_log_probs).reshape(R, A, 1)
        rnn_states = _t2n(rnn_states).reshape(R, A, self.args.recurrent_N, self.args.hidden_size)
        rnn_states_critic = _t2n(rnn_states_critic).reshape(R, A, self.args.recurrent_N, self.args.hidden_size)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    # ------------------------------------------------------------------
    # Insert one step into buffer
    # ------------------------------------------------------------------

    def insert(self, data):
        (
            obs,
            ag_id,
            node_obs,
            adj,
            rewards,
            dones,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        R, A = dones.shape[0], dones.shape[1]

        # masks: 0 when done, 1 otherwise
        masks = (1.0 - dones).reshape(R, A, 1).astype(np.float32)

        # reset rnn states for done agents
        rnn_states[dones.reshape(R, A)] = 0.0
        rnn_states_critic[dones.reshape(R, A)] = 0.0

        share_obs = self._make_share_obs(obs)  # (R, A, 64)
        share_agent_id = self._make_share_agent_id(ag_id)  # (R, A, 8)

        self.buffer.insert(
            obs=obs,
            share_obs=share_obs,
            node_obs=node_obs,
            adj=adj,
            agent_id=ag_id,
            share_agent_id=share_agent_id,
            rnn_states=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards.reshape(R, A, 1).astype(np.float32),
            masks=masks,
        )

    # ------------------------------------------------------------------
    # Compute returns (bootstrap last value)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(self):
        R = self.n_rollout_threads
        A = self.num_agents

        self.trainer.prep_rollout()

        cent_obs = self.buffer.share_obs[-1].reshape(R * A, -1)
        node_obs = self.buffer.node_obs[-1].reshape(R * A, *self.node_obs_shape)
        adj = self.buffer.adj[-1].reshape(R * A, *self.adj_shape)
        agent_id = self.buffer.agent_id[-1].reshape(R * A, -1)
        rnn_states_critic = self.buffer.rnn_states_critic[-1].reshape(
            R * A, self.args.recurrent_N, self.args.hidden_size
        )
        masks = self.buffer.masks[-1].reshape(R * A, -1)

        next_values = self.policy.get_values(cent_obs, node_obs, adj, agent_id, rnn_states_critic, masks)
        next_values = _t2n(next_values).reshape(R, A, 1)

        self.buffer.compute_returns(
            next_values,
            value_normalizer=self.trainer.value_normalizer,
        )

    # ------------------------------------------------------------------
    # Training pass
    # ------------------------------------------------------------------

    def train(self):
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_info

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, episode: int):
        save_dir = os.path.join("checkpoints", self.args.title)
        self.policy.save(save_dir, episode)

    # ------------------------------------------------------------------
    # Extract env metrics from infos
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_infos(infos) -> Dict[str, float]:
        """
        infos: tuple of dicts, one per rollout thread.
        Each dict may have keys: goals_collected, goals_percentage, seen_percentage
        when the episode has ended (auto-reset returns new obs but info reflects final step).
        """
        goals_collected = []
        goals_pct = []
        seen_pct = []
        for info in infos:
            if isinstance(info, dict):
                if "goals_collected" in info:
                    goals_collected.append(info["goals_collected"])
                if "goals_percentage" in info:
                    goals_pct.append(info["goals_percentage"])
                if "seen_percentage" in info:
                    seen_pct.append(info["seen_percentage"])
        return {
            "goals_collected": float(np.mean(goals_collected)) if goals_collected else 0.0,
            "goals_percentage": float(np.mean(goals_pct)) if goals_pct else 0.0,
            "seen_percentage": float(np.mean(seen_pct)) if seen_pct else 0.0,
        }

    # ------------------------------------------------------------------
    # Helper: build share_obs and share_agent_id
    # ------------------------------------------------------------------

    def _make_share_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (R, A, obs_dim)
        Returns (R, A, A*obs_dim): each agent sees all agents' obs concatenated.
        """
        R, A, D = obs.shape
        share = obs.reshape(R, 1, A * D)  # (R, 1, 64)
        return np.repeat(share, A, axis=1)  # (R, A, 64)

    def _make_share_agent_id(self, ag_id: np.ndarray) -> np.ndarray:
        """
        ag_id: (R, A, 1)  int32  values 0..A-1
        Returns (R, A, A): each agent sees all agent IDs concatenated.
        """
        R, A, _ = ag_id.shape
        share = ag_id.reshape(R, 1, A)  # (R, 1, 8)
        return np.repeat(share, A, axis=1)  # (R, A, 8)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self):
        self.warmup()

        start_time = time.time()
        env_metrics: Dict[str, float] = {
            "goals_collected": 0.0,
            "goals_percentage": 0.0,
            "seen_percentage": 0.0,
        }

        for episode in range(self.episodes):
            if self.args.use_linear_lr_decay:
                self.policy.lr_decay(episode, self.episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # actions: (R, A, 1) float32 → int for env
                actions_env = actions.squeeze(-1).astype(np.int32)  # (R, A)

                obs, ag_id, node_obs, adj, rewards, dones, infos = self.envs.step(actions_env)
                # obs:      (R, A, 8)
                # ag_id:    (R, A, 1)
                # node_obs: (R, A, 51, 9)
                # adj:      (R, A, 51, 51)
                # rewards:  (R, A)
                # dones:    (R, A)

                step_metrics = self._extract_infos(infos)
                # Accumulate env metrics (only filled when episode ends in env)
                for k, v in step_metrics.items():
                    if v > 0.0:
                        env_metrics[k] = v

                self.insert(
                    (
                        obs,
                        ag_id,
                        node_obs,
                        adj,
                        rewards,
                        dones,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )
                )

            self.compute()
            train_info = self.train()

            # Logging
            if episode % self.log_interval == 0:
                elapsed = time.time() - start_time
                fps = (episode + 1) * self.episode_length * self.n_rollout_threads / elapsed
                print(
                    f"[Ep {episode:5d}/{self.episodes}] "
                    f"VLoss={train_info['value_loss']:.4f}  "
                    f"PLoss={train_info['policy_loss']:.4f}  "
                    f"Ent={train_info['dist_entropy']:.4f}  "
                    f"Goals={env_metrics['goals_collected']:.1f}  "
                    f"Goals%={env_metrics['goals_percentage']:.2f}  "
                    f"FPS={fps:.0f}"
                )
                self.logger.log_metrics(
                    episode=episode,
                    value_loss=train_info["value_loss"],
                    policy_loss=train_info["policy_loss"],
                    dist_entropy=train_info["dist_entropy"],
                    actor_grad_norm=train_info["actor_grad_norm"],
                    critic_grad_norm=train_info["critic_grad_norm"],
                    goals_collected=env_metrics["goals_collected"],
                    goals_percentage=env_metrics["goals_percentage"],
                    seen_percentage=env_metrics["seen_percentage"],
                    fps=fps,
                )

            if episode % self.save_interval == 0:
                self.save(episode)

        self.logger.finish()
