from __future__ import annotations

import math
import numpy as np
import gymnasium as gym

from abc import abstractmethod
from minigrid.grid import Grid
from typing import Iterable, TypeVar
from minigrid.world import Goal, Obstacle
from utils import Actions, COLOR_NAMES, TILE_PIXELS

T = TypeVar("T")

# Number of node features per entity:
#   rel_pos (2) + 3 * rel_goal_pos (6) + entity_type (1) = 9
NODE_FEAT_DIM = 9


class MultiGridEnv(gym.Env):
    def __init__(self, args, agents) -> None:
        self.clock = None
        self.window = None
        self.agents = agents
        self.goals = []
        self.obstacles = []
        self.seed = args.seed
        self.actions = Actions
        self.screen_size = 640
        self.render_size = None
        self.gamma = args.gamma

        # Environment configuration
        self.entities = []
        self.step_count = 0
        self.num_collected = 0
        self.distance_matrix = np.zeros((1, 1), dtype=np.float32)
        self.width = args.grid_size
        self.height = args.grid_size
        self.see_through_walls = True
        self.num_goals = args.num_goals
        self.max_steps = args.episode_length
        self.num_obstacles = args.num_obstacles
        self.max_edge_dist = args.max_edge_dist
        self.full_features = False  # Use compact 9-feature node obs
        self.grid = Grid(self.width, self.height)

        # Rendering attributes
        self.highlight = True
        self.tile_size = TILE_PIXELS
        self.window_name = "Custom MiniGrid"

        # Dummy goal for missing slots (cur_pos=(0,0) avoids inf in obs)
        self.dummy_goal = Goal()
        self.dummy_goal.cur_pos = (0, 0)
        self.dummy_goal.collected = False

        # Reward values
        self.reward_goal = args.reward_goal
        self.penalty_obstacle = args.penalty_obstacle
        self.penalty_goal = args.penalty_goal
        self.penalty_invalid_move = args.penalty_invalid_move

        self.seen_cells = np.zeros((self.width, self.height), dtype=bool)
        self.total_cells = self.width * self.height - 2 * self.width - 2 * self.height + 4

        # Derived sizes
        num_agents = args.num_agents
        num_goals = args.num_goals
        num_entities = num_agents + num_goals  # obstacles excluded (num_obstacles=0)

        # Gym Space attributes (required by env_wrappers)
        obs_dim = 8  # agent_pos(2) + goal1_pos(2) + goal2_pos(2) + goal3_pos(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.share_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_agents * obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.node_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_entities, NODE_FEAT_DIM), dtype=np.float32
        )
        self.adj_observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(num_entities, num_entities), dtype=np.float32
        )
        self.edge_observation_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        self.agent_id_observation_space = gym.spaces.Box(low=0, high=num_agents - 1, shape=(1,), dtype=np.int32)
        self.share_agent_id_observation_space = gym.spaces.Box(
            low=0, high=num_agents - 1, shape=(num_agents,), dtype=np.int32
        )

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        super().reset(seed=int(np.random.randint(0, 2**31)))
        self.goals = []
        self.entities = []
        self.obstacles = []
        self.dummy_goal = Goal()
        self.dummy_goal.cur_pos = (0, 0)
        self.dummy_goal.collected = False
        self.seen_cells.fill(False)
        self.step_count = 0
        self.num_collected = 0

        for agent in self.agents:
            agent.reset()

        self._gen_grid(self.width, self.height)
        self.calc_distances(True)

        batch_obs = np.stack([self._get_obs(agent) for agent in self.agents])
        batch_node_obs = np.stack([self._get_node_features(agent) for agent in self.agents])
        batch_adj = np.stack([self.distance_matrix for _ in self.agents])

        return batch_obs, batch_node_obs, batch_adj

    @abstractmethod
    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        num_agents = len(self.agents)

        if num_agents == 1:
            x, y = width // 2, height - 2
            self._place_agent(self.agents[0], x, y)
        elif num_agents == 2:
            self._place_agent(self.agents[0], 1, height - 2)
            self._place_agent(self.agents[1], width - 2, height - 2)
        elif num_agents == 3:
            self._place_agent(self.agents[0], 1, height - 2)
            self._place_agent(self.agents[1], width - 2, height - 2)
            self._place_agent(self.agents[2], width // 2, 1)
        elif num_agents == 4:
            self._place_agent(self.agents[0], 1, 1)
            self._place_agent(self.agents[1], width - 2, 1)
            self._place_agent(self.agents[2], 1, height - 2)
            self._place_agent(self.agents[3], width - 2, height - 2)
        else:
            positions = self._get_edge_positions(width, height, num_agents)
            for agent, pos in zip(self.agents, positions):
                self._place_agent(agent, pos[0], pos[1])

        # Place goals (appended to entities after agents)
        for _ in range(self.num_goals):
            obj = Goal()
            self._place_object(obj)
            self.entities.append(obj)

        # num_obstacles == 0 in standard config

    def _place_agent(self, agent, x: int, y: int) -> None:
        agent.init_pos = (x, y)
        agent.cur_pos = (x, y)
        agent.direction = 3
        self.grid.set_agent(x, y, agent)
        self.entities.append(agent)

    def _place_object(self, obj) -> None:
        while True:
            x, y = self._rand_pos(1, self.width - 2, 1, self.height - 2)
            if not self.grid.get(x, y):
                self.grid.set(x, y, obj)
                obj.cur_pos = (x, y)
                if isinstance(obj, Goal):
                    self.goals.append(obj)
                elif isinstance(obj, Obstacle):
                    self.obstacles.append(obj)
                break

    def _get_edge_positions(self, width: int, height: int, num_agents: int) -> list:
        positions = []
        total_edge_length = 2 * (width - 2) + 2 * (height - 2)
        spacing = total_edge_length / num_agents

        current_pos = 0
        for i in range(num_agents):
            edge_pos = int(current_pos)
            if edge_pos < width - 2:
                positions.append((edge_pos + 1, 1))
            elif edge_pos < width + height - 4:
                positions.append((width - 2, edge_pos - width + 3))
            elif edge_pos < 2 * width + height - 6:
                positions.append((2 * width + height - 7 - edge_pos, height - 2))
            else:
                positions.append((1, total_edge_length - edge_pos))
            current_pos += spacing

        return positions

    def _get_obs(self, agent) -> np.ndarray:
        obs = []
        obs.extend(agent.cur_pos)

        agent_id = agent.id
        num_agents = len(self.agents)
        num_goals = len(self.goals)

        # Find goals within FOV
        goals_in_fov = []
        for goal_id in range(num_goals):
            goal = self.goals[goal_id]
            dist = self.distance_matrix[agent_id, num_agents + goal_id]
            collected = self.goals[goal_id].collected

            if dist <= self.max_edge_dist and not collected:
                goals_in_fov.append((goal, dist))

        goals_in_fov.sort(key=lambda x: x[1])
        goals_in_fov = goals_in_fov[:3]

        # Fill remaining with dummy goals (cur_pos=(0,0))
        while len(goals_in_fov) < 3:
            goals_in_fov.append((self.dummy_goal, 1e9))

        # Check if tracked goals have been collected
        if agent.goal1 is not None and agent.goal1.collected:
            agent.goal1 = self.dummy_goal
        if agent.goal2 is not None and agent.goal2.collected:
            agent.goal2 = self.dummy_goal
        if agent.goal3 is not None and agent.goal3.collected:
            agent.goal3 = self.dummy_goal

        if agent.goal1 is None and agent.goal2 is None and agent.goal3 is None:
            if len(goals_in_fov) > 0:
                agent.goal1 = goals_in_fov[0][0]
            if len(goals_in_fov) > 1:
                agent.goal2 = goals_in_fov[1][0]
            if len(goals_in_fov) > 2:
                agent.goal3 = goals_in_fov[2][0]
        elif agent.goal1 is not None and agent.goal2 is not None and agent.goal3 is not None:
            for goal, dist in goals_in_fov:
                if goal == agent.goal1 or goal == agent.goal2 or goal == agent.goal3:
                    continue
                if (
                    agent.goal1 == self.dummy_goal
                    or dist < self.distance_matrix[agent_id, num_agents + self.goals.index(agent.goal1)]
                ):
                    agent.goal3 = agent.goal2
                    agent.goal2 = agent.goal1
                    agent.goal1 = goal
                elif (
                    agent.goal2 == self.dummy_goal
                    or dist < self.distance_matrix[agent_id, num_agents + self.goals.index(agent.goal2)]
                ):
                    agent.goal3 = agent.goal2
                    agent.goal2 = goal
                elif (
                    agent.goal3 == self.dummy_goal
                    or dist < self.distance_matrix[agent_id, num_agents + self.goals.index(agent.goal3)]
                ):
                    agent.goal3 = goal
        else:
            if agent.goal1 is None:
                agent.goal1 = self.dummy_goal
            if agent.goal2 is None:
                agent.goal2 = self.dummy_goal
            if agent.goal3 is None:
                agent.goal3 = self.dummy_goal

        # Relative offsets from agent position → translation-invariant directional signal.
        # (0, 0) when no goal is tracked (dummy or None).
        agent_pos = agent.cur_pos
        for g in [agent.goal1, agent.goal2, agent.goal3]:
            if g is None or g is self.dummy_goal:
                obs.extend([0, 0])
            else:
                obs.extend([g.cur_pos[0] - agent_pos[0], g.cur_pos[1] - agent_pos[1]])

        result = np.array(obs, dtype=np.float32)
        result = np.clip(result, -1e6, 1e6)
        agent.obs = result
        return result

    def _get_node_features(self, agent) -> np.ndarray:
        features = []
        agent_pos = agent.cur_pos

        for entity in self.entities:
            entity_features = []
            entity_pos = entity.cur_pos
            rel_pos = [entity_pos[0] - agent_pos[0], entity_pos[1] - agent_pos[1]]
            entity_features.extend(rel_pos)  # 2 features

            if entity.type == "agent":
                other_agent_goals = [entity.goal1, entity.goal2, entity.goal3]
                for goal in other_agent_goals:
                    if goal is not None:
                        rel_goal_pos = [goal.cur_pos[0] - agent_pos[0], goal.cur_pos[1] - agent_pos[1]]
                    else:
                        rel_goal_pos = [0, 0]
                    entity_features.extend(rel_goal_pos)  # 2 features each
            else:
                # For non-agents, repeat rel_pos 3 times
                for _ in range(3):
                    entity_features.extend(rel_pos)  # 2 features each

            # Entity type: 0=agent, 1=goal, 2=obstacle
            entity_features.append(0 if entity.type == "agent" else (1 if entity.type == "goal" else 2))
            features.append(entity_features)

        result = np.array(features, dtype=np.float32)  # shape (num_entities, 9)
        return result

    def _update_seen_cells(self) -> None:
        for i in range(self.width):
            for j in range(self.height):
                if 0 < i < self.width - 1 and 0 < j < self.height - 1:
                    for agent in self.agents:
                        dist = math.sqrt((i - agent.cur_pos[0]) ** 2 + (j - agent.cur_pos[1]) ** 2)
                        if dist <= self.max_edge_dist:
                            self.seen_cells[i, j] = True

    def _calculate_seen_percentage(self) -> float:
        return np.sum(self.seen_cells) / self.total_cells * 100

    def _handle_overlap(self, i: int, fwd_pos, fwd_cell) -> float:
        if isinstance(fwd_cell, Goal) and not fwd_cell.collected:
            reward = self._reward(self.reward_goal)
            fwd_cell.color = "grey"
            fwd_cell.collected = True
            self.num_collected += 1
        elif isinstance(fwd_cell, Goal) and fwd_cell.collected:
            reward = self._reward(self.penalty_goal)
        elif isinstance(fwd_cell, Obstacle):
            reward = self._reward(self.penalty_obstacle)
        else:
            print(f"fwd_cell = {fwd_cell}")
            raise ValueError("_handle_overlap error.")

        self.grid.set_agent(*fwd_pos, self.agents[i])
        self.grid.set_agent(*self.agents[i].cur_pos, None)
        self.agents[i].cur_pos = fwd_pos
        return reward

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        self.step_count += 1
        num_agents = len(self.agents)
        dones = np.zeros(num_agents, dtype=np.bool_)
        rewards = np.zeros(num_agents, dtype=np.float32)

        for i, action in enumerate(actions):
            action = int(action)
            agent_pos = self.agents[i].cur_pos

            if action == self.actions.up:
                new_pos = (agent_pos[0], agent_pos[1] - 1)
            elif action == self.actions.down:
                new_pos = (agent_pos[0], agent_pos[1] + 1)
            elif action == self.actions.left:
                new_pos = (agent_pos[0] - 1, agent_pos[1])
            elif action == self.actions.right:
                new_pos = (agent_pos[0] + 1, agent_pos[1])
            else:
                new_pos = agent_pos

            cell = self.grid.get(*new_pos)
            agent = self.grid.get_agent(*new_pos)

            if cell is None and agent is None and 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
                self.grid.set_agent(*new_pos, self.agents[i])
                self.grid.set_agent(*self.agents[i].cur_pos, None)
                self.agents[i].cur_pos = new_pos
            elif cell and cell.can_overlap():
                rewards[i] = self._handle_overlap(i, new_pos, cell)
            else:
                rewards[i] = self._reward(self.penalty_invalid_move)

        episode_done = self.step_count >= self.max_steps or self.num_collected >= self.num_goals

        if episode_done:
            dones[:] = True

        self.calc_distances(False)
        self._update_seen_cells()
        seen_percentage = self._calculate_seen_percentage()

        if episode_done:
            info = {
                "goals_collected": self.num_collected,
                "goals_percentage": (self.num_collected / self.num_goals) * 100,
                "seen_percentage": seen_percentage,
            }
            # Auto-reset: regenerate grid and return new episode's first obs
            self._auto_reset()
        else:
            info = {}

        batch_obs = np.stack([self._get_obs(agent) for agent in self.agents])
        batch_node_obs = np.stack([self._get_node_features(agent) for agent in self.agents])
        batch_adj = np.stack([self.distance_matrix for _ in self.agents])

        return batch_obs, batch_node_obs, batch_adj, rewards, dones, info

    def _auto_reset(self) -> None:
        self.goals = []
        self.entities = []
        self.obstacles = []
        self.dummy_goal = Goal()
        self.dummy_goal.cur_pos = (0, 0)
        self.dummy_goal.collected = False
        self.seen_cells.fill(False)
        self.step_count = 0
        self.num_collected = 0

        for agent in self.agents:
            agent.reset()

        self._gen_grid(self.width, self.height)
        self.calc_distances(True)

    def calc_distances(self, reset: bool = False) -> None:
        num_entities = len(self.entities)
        num_agents = len(self.agents)

        if reset:
            distance_matrix = np.zeros((num_entities, num_entities), dtype=np.float32)
            for i in range(num_entities):
                for j in range(i + 1, num_entities):
                    x1, y1 = self.entities[i].cur_pos
                    x2, y2 = self.entities[j].cur_pos
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        else:
            distance_matrix = self.distance_matrix.copy()
            for i in range(num_agents):
                for j in range(num_agents, num_entities):
                    x1, y1 = self.agents[i].cur_pos
                    x2, y2 = self.entities[j].cur_pos
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

        self.distance_matrix = distance_matrix

    def render_env(self) -> None:
        import pygame

        img = self.get_full_render(self.highlight, self.tile_size)
        img = np.transpose(img, axes=(1, 0, 2))

        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface((int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset)))
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))
        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        pygame.display.flip()

    def get_full_render(self, highlight: bool, tile_size: int) -> np.ndarray:
        highlight_masks = np.zeros((self.width, self.height), dtype=bool)

        for i in range(self.width):
            for j in range(self.height):
                for agent in self.agents:
                    dist = math.sqrt((i - agent.cur_pos[0]) ** 2 + (j - agent.cur_pos[1]) ** 2)
                    if dist <= self.max_edge_dist:
                        highlight_masks[i, j] = True

        img = self.grid.render(tile_size, highlight_mask=highlight_masks if highlight else None)
        return img

    def _reward(self, reward: float) -> float:
        return reward * (self.gamma**self.step_count)

    def _rand_int(self, low: int, high: int) -> int:
        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list:
        lst = list(iterable)
        assert num_elems <= len(lst)
        out = []
        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)
        return out

    def _rand_color(self) -> str:
        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, x_low: int, x_high: int, y_low: int, y_high: int) -> tuple:
        return self.np_random.integers(x_low, x_high), self.np_random.integers(y_low, y_high)

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
