import numpy as np

from typing import Any
from utils import TILE_PIXELS
from minigrid.world import Wall
from minigrid.rendering import downsample, highlight_img, fill_coords, point_in_rect


class Grid:
    tile_cache: dict[tuple[Any, ...], Any] = {}
    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3
        self.width = width
        self.height = height
        self.grid = [None] * (width * height)
        self.agent_grid = [None] * (width * height)

    def set(self, i, j, v):
        assert (0 <= i < self.width), f"column index {i} outside of grid of width {self.width}"
        assert (0 <= j < self.height), f"row index {j} outside of grid of height {self.height}"
        self.grid[j * self.width + i] = v

    def get(self, i: int, j: int):
        assert 0 <= i <= self.width
        assert 0 <= j <= self.height
        assert self.grid is not None
        return self.grid[j * self.width + i]

    def set_agent(self, i, j, v):
        assert (0 <= i < self.width), f"column index {j} outside of grid of width {self.width}"
        assert (0 <= j < self.height), f"row index {j} outside of grid of height {self.height}"
        if v is None:
            self.agent_grid[j * self.width + i] = None
        else:
            if self.agent_grid[j * self.width + i] is None:
                self.agent_grid[j * self.width + i] = [v]
            else:
                self.agent_grid[j * self.width + i].append(v)

    def get_agent(self, i, j):
        assert (0 <= i < self.width), f"column index {j} outside of grid of width {self.width}"
        assert (0 <= j < self.height), f"row index {j} outside of grid of height {self.height}"
        assert self.agent_grid is not None
        return self.agent_grid[j * self.width + i]

    def horz_wall(self, x, y, length=None,obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    @classmethod
    def render_tile(cls, obj=None, agent=None, highlights=False, tile_size=TILE_PIXELS, subdivs=3):
        key: tuple[Any, ...] = (highlights, tile_size)
        key = obj.encode() + key if obj else key
        key = agent.encode() + key if agent else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)
        if agent != None:
            agent.render(img)
        if highlights:
            highlight_img(img)

        img = downsample(img, subdivs)
        cls.tile_cache[key] = img
        return img

    def render(self, tile_size, highlight_mask=None):
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                agents = self.get_agent(i, j)
                tile_img = Grid.render_tile(cell, highlights=highlight_mask[i, j], tile_size=tile_size)
                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
                if agents is not None:
                    for agent in agents:
                        agent_img = Grid.render_tile(agent, highlights=highlight_mask[i, j], tile_size=tile_size)
                        img[ymin:ymax, xmin:xmax, :] = agent_img
        return img
