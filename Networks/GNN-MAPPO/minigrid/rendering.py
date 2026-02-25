import numpy as np
from typing import Callable


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0
    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)
    return img


def fill_coords(img: np.ndarray, fn: Callable[[float, float], bool], color) -> np.ndarray:
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def point_in_circle(cx: float, cy: float, r: float) -> Callable[[float, float], bool]:
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(x_min: float, x_max: float, y_min: float, y_max: float) -> Callable[[float, float], bool]:
    def fn(x, y):
        return x_min <= x <= x_max and y_min <= y <= y_max

    return fn


def highlight_img(img: np.ndarray, color: tuple = (255, 255, 255), alpha: float = 0.30) -> None:
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
