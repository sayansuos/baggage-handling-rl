import math
import os

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import Experiment
from simulator.environment.environment import Environment


def plot_figures(
    df_perf: pd.DataFrame,
    df_rewards: pd.DataFrame,
    path: str | None,
    file_name: str | None,
    window: int = 10,
    render: bool = False,
):
    """Generate and save the reward, performance, and velocity plots for an experiment."""

    plot_rewards(df_rewards, path, file_name, window, render)
    plot_performances(df_perf, path, file_name, window, render)
    plot_velocities(df_perf, path, file_name, window, render)


def plot_renders(
    experiments: list[Experiment],
    path: str,
    file_name: str = "",
    min_ncols: int = 2,
):
    """Render the initial environment of each experiment and save them in a single comparison figure."""

    os.makedirs(path, exist_ok=True)

    n = len(experiments)
    ncols = min(min_ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3 * nrows),
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, exp in zip(axes, experiments):

        env = Environment(
            env_config=exp.env_config,
            agent_config=exp.agent_config,
            reward_config=exp.reward_config,
            name=exp.name,
        )
        env.reset(1234)

        env.render(ax=ax)
        ax.set_title(exp.name)
        ax.tick_params(axis="both", labelsize=6)

    for ax in axes[len(experiments) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{path}/{file_name}_renders.png", dpi=300)
    plt.close(fig)


def plot_grid(
    grid: np.ndarray,
    path: str,
    file_name: str = "",
    scale: int = 10,
):
    """Save an occupancy grid as an image, with free cells in white and occupied cells in black."""

    os.makedirs(path, exist_ok=True)

    img = (1 - grid) * 255
    H, W = img.shape

    img = cv2.resize(
        img.astype(np.uint8),
        (W * scale, H * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for x in range(0, W * scale, scale):
        cv2.line(img, (x, 0), (x, H * scale), (200, 200, 200), 1)
    for y in range(0, H * scale, scale):
        cv2.line(img, (0, y), (W * scale, y), (200, 200, 200), 1)

    imageio.imwrite(f"{path}/{file_name}_grid.png", img)


def plot_animation(frames, path: str, file_name: str = "", fps: int = 20):
    """Save a sequence of rendered frames as both an MP4 video and a GIF animation."""

    os.makedirs(path, exist_ok=True)

    mp4_file = f"{path}/{file_name}_anim.mp4"
    writer = imageio.get_writer(mp4_file, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    gif_file = f"{path}/{file_name}_anim.gif"
    imageio.mimsave(
        gif_file,
        frames,
        fps=min(fps, 10),
        loop=0,
    )


def plot_rewards(
    df: pd.DataFrame,
    path: str | None,
    file_name: str | None,
    window: int = 10,
    render: bool = False,
):
    """Plot the smoothed evolution of the total reward and its different components over episodes."""

    for col in [
        "return_total",
        "reward_progress",
        "reward_collision",
        "reward_safety",
        "reward_rotation",
    ]:
        df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        df["episode"],
        df["reward_progress_smooth"],
        alpha=0.3,
        label="Progress",
    )

    ax.plot(
        df["episode"],
        df["reward_collision_smooth"],
        alpha=0.3,
        label="Collision",
    )

    ax.plot(
        df["episode"],
        df["reward_safety_smooth"],
        alpha=0.3,
        label="Safety",
    )

    ax.plot(
        df["episode"],
        df["reward_rotation_smooth"],
        alpha=0.3,
        label="Rotation",
    )

    ax.plot(
        df["episode"],
        df["return_total_smooth"],
        color="black",
        label="Total",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward decomposition")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if path and file_name:
        fig.savefig(f"{path}/{file_name}_rewards.png", dpi=300)

    if render:
        plt.show()

    plt.close(fig)


def plot_performances(
    df: pd.DataFrame,
    path: str | None,
    file_name: str | None,
    window: int = 10,
    render: bool = False,
):
    """Plot the smoothed success, collision, timeout rates, and mean travel time over episodes."""

    df["timeout_rate"] = 1 - df["success_rate"] - df["collision_rate"]

    for col in ["success_rate", "collision_rate", "timeout_rate", "mean_time_travel"]:
        df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
    )

    # Plot rates

    ax1.plot(
        df["episode"],
        df["success_rate_smooth"],
        linewidth=2,
        label="Success rate",
    )

    ax1.plot(
        df["episode"],
        df["collision_rate_smooth"],
        linewidth=2,
        label="Collision rate",
    )

    ax1.plot(
        df["episode"],
        df["timeout_rate_smooth"],
        linewidth=2,
        label="Timeout rate",
    )

    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Training performance")

    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot time travel

    ax2.plot(
        df["episode"],
        df["mean_time_travel_smooth"],
        linewidth=2,
        label="Mean travel time",
    )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")

    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    if path and file_name:
        fig.savefig(
            f"{path}/{file_name}_performances.png",
            dpi=300,
        )

    if render:
        plt.show()

    plt.close(fig)


def plot_velocities(
    df: pd.DataFrame,
    path: str | None,
    file_name: str | None,
    window: int = 10,
    render: bool = False,
):
    """Plot the smoothed mean linear and angular velocities over episodes."""

    for col in ["mean_v", "mean_abs_omega"]:
        df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        df["episode"], df["mean_v_smooth"], linewidth=1, label="Mean v", color="blue"
    )

    ax.plot(
        df["episode"],
        df["mean_abs_omega_smooth"],
        linewidth=1,
        label="Mean |omega|",
        color="orange",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("Linear and angular velocities")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if path and file_name:
        fig.savefig(
            f"{path}/{file_name}_velocities.png",
            dpi=300,
        )

    if render:
        plt.show()

    plt.close(fig)
