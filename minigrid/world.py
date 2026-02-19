from __future__ import annotations
import numpy as np

from typing import Tuple
from minigrid.rendering import fill_coords, point_in_circle, point_in_rect
from utils import COLOR_TO_IDX, COLORS, IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX, DIR_TO_VEC

Point = Tuple[int, int]


class WorldObj:
    def __init__(self, type: str, color: str, id=None) -> None:
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.init_pos = None
        self.cur_pos = None
        self.color = color
        self.type = type
        self.id = id

    def can_overlap(self) -> bool:
        return False

    def encode(self) -> tuple:
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0

    @staticmethod
    def decode(embedding) -> tuple:
        obj_type = IDX_TO_OBJECT[embedding[0]]
        color = IDX_TO_COLOR[embedding[1]]

        if obj_type == "unseen":
            return None, None
        if obj_type == "empty":
            v = None
        elif obj_type == "wall":
            v = Wall(color)
        elif obj_type == "goal":
            v = Goal(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        w = Agent.decode(*embedding[3:])

        return v, w


class Agent(WorldObj):
    def __init__(self, id: int, args) -> None:
        super(Agent, self).__init__("agent", "purple")
        self.id = id
        self.direction = 3
        self.color = "purple"
        self.num_collected = 0
        self.num_goals = args.num_goals
        self.num_agents = args.num_agents
        self.num_entities = args.num_agents + args.num_goals
        self.obs = None
        self.node_obs = None
        self.goal1 = None
        self.goal2 = None
        self.goal3 = None

    def render(self, img: np.ndarray) -> None:
        c = COLORS[self.color]
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), c)

    def encode(self) -> tuple:
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.direction, 0, 0, 0

    @staticmethod
    def decode(type, color_index, direction) -> Agent | None:
        obj_type = IDX_TO_OBJECT[type]
        if obj_type == "empty":
            v = None
        elif obj_type == "agent":
            v = Agent(color_index=color_index, direction=direction)
        else:
            assert False, "unknown object type in decode '%s'" % type
        return v

    def reset(self) -> None:
        self.direction = 3
        self.num_collected = 0
        self.obs = None
        self.node_obs = None
        self.goals = []
        self.goal1 = None
        self.goal2 = None
        self.goal3 = None

    @property
    def dir_vec(self) -> np.ndarray:
        assert 0 <= self.direction < 4
        return DIR_TO_VEC[3]

    @property
    def right_vec(self) -> np.ndarray:
        dx, dy = self.direction_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self) -> np.ndarray:
        return self.cur_pos + self.direction_vec

    def can_overlap(self) -> bool:
        return False


class Wall(WorldObj):
    def __init__(self, color: str = "grey") -> None:
        super().__init__("wall", color)

    def render(self, img: np.ndarray) -> None:
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, color: str = "green") -> None:
        super().__init__("goal", color)
        self.collected = False

    def can_overlap(self) -> bool:
        return True

    def render(self, img: np.ndarray) -> None:
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Obstacle(WorldObj):
    def __init__(self, color: str = "red") -> None:
        super().__init__("obstacle", color)

    def can_overlap(self) -> bool:
        return True

    def render(self, img: np.ndarray) -> None:
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
