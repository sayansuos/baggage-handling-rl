import os

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_rewards(
    df: pd.DataFrame, path: str, file_name: str, window: int = 10, render: bool = False
):

    for col in [
        "return_total",
        "reward_progress",
        "reward_collision",
        "reward_safety",
        "reward_rotation",
    ]:
        df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

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
    fig.savefig(f"{path}/{file_name}_rewards.png", dpi=300)

    if render:
        plt.show()

    plt.close(fig)


def plot_performances(
    df: pd.DataFrame, path: str, file_name: str, window: int = 10, render: bool = False
):
    df["timeout_rate"] = 1 - df["success_rate"] - df["collision_rate"]

    for col in ["success_rate", "collision_rate", "timeout_rate", "mean_time_travel"]:
        df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
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
    fig.savefig(
        f"{path}/{file_name}_performances.png",
        dpi=300,
    )

    if render:
        plt.show()

    plt.close(fig)


def plot_velocities(
    df: pd.DataFrame, path: str, file_name: str, window: int = 10, render: bool = False
):
    for col in ["mean_v", "mean_abs_omega"]:
        df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df["episode"],
        df["mean_v"],
        alpha=0.3,
        linewidth=1,
        label="Mean v (raw)",
        color="blue",
    )

    ax.plot(
        df["episode"],
        df["mean_abs_omega"],
        alpha=0.3,
        linewidth=1,
        label="Mean |omega| (raw)",
        color="orange",
    )

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
    fig.savefig(
        f"{path}/{file_name}_velocities.png",
        dpi=300,
    )

    if render:
        plt.show()

    plt.close(fig)


def plot_grid(
    grid: np.ndarray,
    path: str,
    file_name: str = "",
    scale: int = 10,
):
    """
    Save an occupancy grid as an image.
    """

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
    """
    Save a list of frames as an animation.
    """

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
