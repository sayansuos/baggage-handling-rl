import os

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_animation(frames, path: str, file_name: str = "", fps: int = 20):
    """
    Save a list of frames as an animation.
    """

    os.makedirs(path, exist_ok=True)

    file = f"{path}/{file_name}_anim.mp4"
    writer = imageio.get_writer(file, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


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


def plot_rewards(
    df: pd.DataFrame,
    path: str,
    file_name: str = "",
):
    """
    Save reward and cumulative reward plots from debug logs.
    """

    os.makedirs(path, exist_ok=True)

    experiments = df["environment"].unique()
    fig, axes = plt.subplots(
        nrows=len(experiments),
        ncols=2,
        figsize=(12, 4 * len(experiments)),
        squeeze=False,
    )

    for row, exp in enumerate(experiments):

        # Get infos

        data = df[df["environment"] == exp].copy()
        success_steps = (
            data.loc[data["goal_distance"] < 0.5]
            .groupby("agent", as_index=False)["step"]
            .min()
        )["step"].tolist()
        collision_steps = (
            data.loc[data["state"] == "truncated"]
            .groupby("agent", as_index=False)["step"]
            .min()
        )["step"].tolist()
        exp_df = (
            data.groupby("step", as_index=False)
            .agg(
                reward=("reward", "sum"),
            )
            .sort_values("step")
        )
        steps = exp_df["step"]
        rewards = exp_df["reward"]

        # Plot

        ax_reward = axes[row, 0]
        ax_cumsum = axes[row, 1]

        for agent in data["agent"].unique():
            agent_df = data[data["agent"] == agent].sort_values("step")
            ax_reward.plot(
                agent_df["step"],
                agent_df["reward"],
                alpha=0.3,
                label=agent,
                linestyle="--",
            )
            ax_cumsum.plot(
                agent_df["step"],
                agent_df["reward"].cumsum(),
                alpha=0.3,
                label=agent,
                linestyle="--",
            )

        ax_reward.plot(steps, rewards, color="black")
        for s in collision_steps:
            ax_reward.axvline(s, color="red", alpha=0.7)
        for s in success_steps:
            ax_reward.axvline(s, color="green", alpha=0.7)
        ax_reward.set_title(f"Reward Function | {exp}")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.legend()
        ax_reward.grid(True)

        ax_cumsum.plot(steps, rewards.cumsum(), color="black")
        for s in collision_steps:
            ax_cumsum.axvline(s, color="red", alpha=0.7)
        for s in success_steps:
            ax_cumsum.axvline(s, color="green", alpha=0.7)

        ax_cumsum.set_title(f"Cumulative Reward Function | {exp}")
        ax_cumsum.set_xlabel("Step")
        ax_cumsum.legend()
        ax_cumsum.grid(True)

    fig.tight_layout()
    fig.savefig(f"{path}/{file_name}_reward_tracking.png", dpi=300)
    plt.close(fig)


def plot_training_metrics(
    df: pd.DataFrame,
    path: str,
    file_name: str = "",
):
    """
    Save SAC training metrics and plot return evolution.
    """

    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["episode"], df["return"], label="Score", alpha=0.4)
    ax.plot(df["episode"], df["average_return"], label="Average return")
    ax.plot(df["episode"], df["best_return"], label="Best return", linestyle="--")

    ax.set_title(f"SAC training | {file_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"{path}/{file_name}_metrics.png", dpi=300)
    plt.close(fig)

    return df
