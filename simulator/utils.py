import imageio.v2 as imageio
import numpy as np
from simulator.environment.environment import Environment


def save_grid(grid: np.ndarray, file_name: str, path: str = "figures/"):
    imageio.imwrite(path + file_name, (1 - grid) * 255)


def save_animation(
    env: Environment,
    file_name: str,
    path: str = "figures/",
    step: int = 300,
    fps: int = 30,
):
    writer = imageio.get_writer(path + file_name, fps=fps)

    for _ in range(step):
        env.step()
        env.render()
        env.fig.canvas.draw()
        frame = np.asarray(env.fig.canvas.renderer.buffer_rgba())
        writer.append_data(frame)

    writer.close()
