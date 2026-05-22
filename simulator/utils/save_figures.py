import cv2
import imageio.v2 as imageio
import numpy as np
from simulator.environment.environment import Environment


def save_grid(
    grid: np.ndarray,
    file_name: str,
    path: str = "figures/",
    scale: int = 10,
    show_grid: bool = False,
):

    img = (1 - grid) * 255
    H, W = img.shape

    img = cv2.resize(
        img,
        (W * scale, H * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if show_grid:
        for x in range(0, W * scale, scale):
            cv2.line(img, (x, 0), (x, H * scale), (200, 200, 200), 1)
        for y in range(0, H * scale, scale):
            cv2.line(img, (0, y), (W * scale, y), (200, 200, 200), 1)

    imageio.imwrite(path + file_name, img)
