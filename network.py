import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from torch import Tensor
from typing import Optional

from gnn import GNNBase
from collections import deque, namedtuple


class GR_QNetwork:
    def __init__(self, args) -> None:
        self.t_step = 0
        self.lr = args.lr
        self.tau = args.tau
        self.seed = args.seed
        self.gamma = args.gamma
        if args.priority_replay:
            self.prio_e = args.prio_e
            self.prio_a = args.prio_a
            self.prio_b = args.prio_b
        self.num_agents = args.num_agents
        self.double_dqn = args.double_dqn
        self.batch_size = args.batch_size
        self.action_size = args.action_size
        self.update_step = args.update_step
        self.buffer_size = args.buffer_size
        self.device = torch.device(args.device)

        # Q-Network + Buffer
        self.qnetwork_local = GNNBase(args).to(self.device)
        # self.qnetwork_local.count_layers_and_params()
        self.qnetwork_target = GNNBase(args).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args)

    @property
    def time_step(self) -> int:
        return self.t_step // self.num_agents

    def step(
        self,
        agent_id: Tensor,
        obs: Tensor,
        node_obs: Tensor,
        adj: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        next_node_obs: Tensor,
        next_adj: Tensor,
        done: Tensor,
    ) -> Optional[Tensor]:
        loss = None
        self.t_step += 1

        # Add experience to replay buffer - using proper tensor cloning
        self.memory.add(
            agent_id.clone().detach(),
            obs.float().clone().detach(),
            node_obs.float().clone().detach(),
            adj.float().clone().detach(),
            action.clone().detach(),
            reward.clone().detach(),
            next_obs.float().clone().detach(),
            next_node_obs.float().clone().detach(),
            next_adj.float().clone().detach(),
            done.clone().detach(),
        )

        # Update the network
        if self.time_step % self.update_step == 0:
            if len(self.memory) > self.batch_size:
                experiences, experience_indexes, priorities = self.memory.sample()
                loss = self.learn(experiences, experience_indexes, priorities, self.gamma)
        return loss

    def action(self, obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, eps: float, debug: bool) -> Tensor:
        obs = obs.float().to(self.device)
        node_obs = node_obs.float().to(self.device)
        adj = adj.float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(obs, node_obs, adj, agent_id)
        self.qnetwork_local.train()

        # Handle epsilon-greedy for the whole batch
        actions = []
        for i in range(len(obs)):
            if random.random() > eps:
                action = action_values[i].argmax().item()
            else:
                if debug:
                    # Keep debug input handling for single agent
                    while True:
                        user_input = input("Enter an action (W,A,S,D): ")
                        if user_input == "a":
                            action = 0  # Left
                            break
                        elif user_input == "d":
                            action = 1  # Right
                            break
                        elif user_input == "w":
                            action = 2  # Up
                            break
                        elif user_input == "s":
                            action = 3  # Down
                            break
                        else:
                            print("Invalid input. Please enter (W,A,S,D)")
                else:
                    action = random.randint(0, self.action_size - 1)
            actions.append(action)

        return torch.tensor(actions, device=self.device)

    def learn(self, experiences: tuple, experience_indexes: Tensor, priorities: Tensor, gamma: float) -> Tensor:
        agent_id, obs, node_obs, adj, actions, rewards, next_obs, next_node_obs, next_adj, done = experiences

        # Calculate current Q(s,a) - needs gradients for training
        Q_s = self.qnetwork_local(obs, node_obs, adj, agent_id)
        Q_s_a = Q_s.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target calculation - don't need gradients
        with torch.no_grad():
            next_Q = self.qnetwork_target(next_obs, next_node_obs, next_adj, agent_id)
            Q_s_next = next_Q.max(1)[0]
            targets = rewards + gamma * Q_s_next * (1 - done.float())

        # Loss calculation
        losses = (Q_s_a - targets) ** 2
        importance_weights = (((1 / self.buffer_size) * (1 / priorities)) ** self.prio_b).unsqueeze(1)
        loss = (importance_weights * losses).mean()

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Priority updates - don't need gradients
        with torch.no_grad():
            td_errors = abs(Q_s_a - targets)
            new_priorities = td_errors + self.prio_e
            self.memory.update_priority(experience_indexes, new_priorities)

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_beta(self, current_step: int, total_steps: int, beta_start: float) -> None:
        beta = beta_start + (1.0 - beta_start) * (current_step / total_steps)
        self.prio_b = min(beta, 1.0)


class ReplayBuffer:
    def __init__(self, args) -> None:
        self.action_size = args.action_size
        self.device = torch.device(args.device)
        self.memory = deque(maxlen=args.buffer_size)
        self.priority = deque(maxlen=args.buffer_size)
        self.batch_size = args.batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "id",
                "obs",
                "node_obs",
                "adj",
                "action",
                "reward",
                "next_obs",
                "next_node_obs",
                "next_adj",
                "done",
            ],
        )
        self.seed = args.seed
        if args.priority_replay:
            self.prio_e = args.prio_e
            self.prio_a = args.prio_a
            self.prio_b = args.prio_b

    def add(
        self,
        id: Tensor,
        obs: Tensor,
        node_obs: Tensor,
        adj: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        next_node_obs: Tensor,
        next_adj: Tensor,
        dones: Tensor,
    ) -> None:
        for i in range(len(id)):
            e = self.experience(
                id[i],
                obs[i],
                node_obs[i],
                adj[i],
                action[i],
                reward[i],
                next_obs[i],
                next_node_obs[i],
                next_adj[i],
                dones[i],
            )
            self.memory.append(e)
            self.priority.append(self.prio_e)

    def update_priority(self, priority_indexes: Tensor, priority_targets: Tensor) -> None:
        for index, priority_index in enumerate(priority_indexes):
            self.priority[priority_index] = priority_targets[index].item()

    def sample(self) -> tuple[tuple, Tensor, Tensor]:
        adjusted_priority = torch.tensor(self.priority, dtype=torch.float32, device=self.device) ** self.prio_a
        sampling_probability = adjusted_priority / adjusted_priority.sum()
        experience_indexes = torch.multinomial(sampling_probability, self.batch_size, replacement=False)
        experiences = [self.memory[index] for index in experience_indexes]

        id = torch.stack([e.id for e in experiences if e is not None]).to(self.device)
        obs = torch.stack([e.obs for e in experiences if e is not None]).to(self.device)
        node_obs = torch.stack([e.node_obs for e in experiences if e is not None]).to(self.device)
        adj = torch.stack([e.adj for e in experiences if e is not None]).to(self.device)
        action = torch.stack([e.action for e in experiences if e is not None]).to(self.device)
        reward = torch.stack([e.reward for e in experiences if e is not None]).to(self.device)
        next_obs = torch.stack([e.next_obs for e in experiences if e is not None]).to(self.device)
        next_node_obs = torch.stack([e.next_node_obs for e in experiences if e is not None]).to(self.device)
        next_adj = torch.stack([e.next_adj for e in experiences if e is not None]).to(self.device)
        done = torch.stack([e.done for e in experiences if e is not None]).to(self.device)
        priorities = torch.tensor(
            [self.priority[index] for index in experience_indexes], dtype=torch.float32, device=self.device
        )

        return (
            (id, obs, node_obs, adj, action, reward, next_obs, next_node_obs, next_adj, done),
            experience_indexes,
            priorities,
        )

    def __len__(self) -> int:
        return len(self.memory)
