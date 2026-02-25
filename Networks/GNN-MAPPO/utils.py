import copy
import math
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray as arr
from typing import Tuple, Optional, Union
from enum import IntEnum


TILE_PIXELS = 32

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "orange": np.array([255, 165, 0]),
    "cyan": np.array([0, 255, 255]),
    "magenta": np.array([255, 0, 255]),
    "lime": np.array([0, 255, 0]),
    "pink": np.array([255, 192, 203]),
    "teal": np.array([0, 128, 128]),
    "lavender": np.array([230, 230, 250]),
    "brown": np.array([165, 42, 42]),
    "beige": np.array([245, 245, 220]),
    "maroon": np.array([128, 0, 0]),
    "mint": np.array([192, 255, 192]),
    "olive": np.array([128, 128, 0]),
    "coral": np.array([255, 127, 80]),
    "navy": np.array([0, 0, 128]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
}
COLOR_NAMES = sorted(list(COLORS.keys()))

COLOR_TO_IDX = {
    "red": 1,
    "green": 2,
    "blue": 3,
    "purple": 4,
    "yellow": 5,
    "grey": 6,
    "orange": 7,
    "cyan": 8,
    "magenta": 9,
    "lime": 10,
    "pink": 11,
    "teal": 12,
    "lavender": 13,
    "brown": 14,
    "beige": 15,
    "maroon": 16,
    "mint": 17,
    "olive": 18,
    "coral": 19,
    "navy": 20,
    "white": 21,
    "black": 22,
}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "agent": 3,
    "goal": 4,
    "obstacle": 5,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

DIR_TO_VEC = [
    np.array((1, 0)),
    np.array((0, 1)),
    np.array((-1, 0)),
    np.array((0, -1)),
]


class Actions(IntEnum):
    left = 0
    right = 1
    up = 2
    down = 3


def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def check(input) -> torch.Tensor:
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    return input


def _t2n(x) -> np.ndarray:
    return x.detach().cpu().numpy()


def get_grad_norm(it) -> float:
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def update_linear_schedule(optimizer, epoch: int, total_num_epochs: int, initial_lr: float) -> None:
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape,
        norm_axes: int = 1,
        beta: float = 0.99999,
        per_element_update: bool = False,
        epsilon: float = 1e-5,
        device=torch.device("cpu"),
    ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self) -> Tuple[torch.Tensor, torch.Tensor]:
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: Union[torch.Tensor, arr]):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: Union[torch.Tensor, arr]) -> torch.Tensor:
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        stddev = torch.sqrt(var)[(None,) * self.norm_axes]
        out = (input_vector - mean[(None,) * self.norm_axes]) / stddev
        return out

    def denormalize(self, input_vector: Union[torch.Tensor, arr]) -> np.ndarray:
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out.cpu().numpy()
