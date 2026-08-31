import math
from pathlib import Path
from typing import Literal

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from configs.config import Task
from simulator.environment.environment import Environment


def plot_renders(
    tasks: list[Task],
    path: str | Path,
    file_name: str = "",
    max_ncols: int = 2,
) -> None:
    """
    Render the initial environment of each task and save them in a single
    comparison figure.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Compute the number of rows and columns
    n = len(tasks)
    ncols = min(max_ncols, n)
    nrows = math.ceil(n / ncols)

    # Create the subplot grid
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    # Render the initial state of each task
    for ax, task in zip(axes, tasks):
        # Initialize the environment from the task configuration
        env = Environment(
            env_config=task.env_config,
            agent_config=task.agent_config,
            reward_config=task.reward_config,
            name=task.name,
        )
        env.set_focus_agents(n_focus_agents=env.env_config.nb_agents)
        env.reset(1234)

        # Render the environment
        env.render(ax=ax)
        ax.set_title(task.name)
        ax.tick_params(axis="both", labelsize=6)

    # Hide unused subplots
    for ax in axes[len(tasks) :]:
        ax.axis("off")

    # Save the figure
    plt.tight_layout()
    fig.savefig(path / f"{file_name}_renders.png", dpi=300)
    plt.close(fig)


def plot_grid(
    grid: np.ndarray,
    path: str | Path,
    file_name: str = "",
    scale: int = 10,
) -> None:
    """
    Save an occupancy grid as an image, with free cells in white and occupied cells in
    black.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Convert the binary occupancy grid to grayscale
    img = (1 - grid) * 255

    # Retrieve the grid dimensions
    H, W = img.shape

    # Enlarge the grid
    img = cv2.resize(
        img.astype(np.uint8),
        (W * scale, H * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    # Convert the grayscale image to RGB to allow colored grid lines
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Draw vertical grid lines
    for x in range(0, W * scale, scale):
        cv2.line(img, (x, 0), (x, H * scale), (200, 200, 200), 1)

    # Draw horizontal grid lines
    for y in range(0, H * scale, scale):
        cv2.line(img, (0, y), (W * scale, y), (200, 200, 200), 1)

    # Save the figure
    imageio.imwrite(path / f"{file_name}_grid.png", img)


def plot_animation(frames, path: str, file_name: str = "", fps: int = 20):
    """
    Save a sequence of rendered frames as both an MP4 video and a GIF animation.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Define the output file paths
    mp4_file = path / f"{file_name}_anim.mp4"
    gif_file = path / f"{file_name}_anim.gif"

    # Create the video writer
    writer = imageio.get_writer(mp4_file, fps=fps)

    # Add each rendered frame to the video
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    # Save the frames as a GIF
    imageio.mimsave(gif_file, frames, fps=min(fps, 10), loop=0)


def plot_figures(
    logs_dir: str,
    figs_dir: str,
    mode: Literal["train", "validation", "evaluation"],
    policy_name: str,
    checkpoint_name: str | None,
    window: int,
) -> tuple[Figure, Figure, Figure]:
    """
    Generate training curves or validation/evaluation summaries.

    For training, ''file_name'' identifies the task or curriculum log.
    For validation and evaluation, all task logs of the policy are combined.
    """

    # Create the output directory if it does not exist
    if mode == "train":
        figs_dir = Path(figs_dir) / mode / policy_name
    else:
        figs_dir = Path(figs_dir) / mode / policy_name / checkpoint_name
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load one task logs for train
    if mode == "train":
        logs_dir = Path(logs_dir) / mode / policy_name
        metrics_file = logs_dir / "metrics.csv"
        rewards_file = logs_dir / "rewards.csv"

        if not metrics_file.is_file():
            raise FileNotFoundError(f"Performance log not found: {metrics_file}")
        if not rewards_file.is_file():
            raise FileNotFoundError(f"Reward log not found: {rewards_file}")

        metrics = pd.read_csv(metrics_file)
        rewards = pd.read_csv(rewards_file)

    # Load all tasks logs for validation and evaluation
    else:
        logs_dir = Path(logs_dir) / mode / policy_name / checkpoint_name
        metrics_files = sorted(logs_dir.glob("*_metrics.csv"))
        rewards_files = sorted(logs_dir.glob("*_rewards.csv"))

        if not metrics_files:
            raise FileNotFoundError(f"No metrics files found in: {logs_dir}")
        if not rewards_files:
            raise FileNotFoundError(f"No reward files found in: {logs_dir}")

        # Concatenate all files in one
        metrics_frames = []
        rewards_frames = []

        for metrics_file in metrics_files:
            task_name = metrics_file.name.removesuffix("_metrics.csv")
            task_df = pd.read_csv(metrics_file)
            task_df["task"] = task_name
            metrics_frames.append(task_df)

        for rewards_file in rewards_files:
            task_name = rewards_file.name.removesuffix("_rewards.csv")
            task_df = pd.read_csv(rewards_file)
            task_df["task"] = task_name
            rewards_frames.append(task_df)

        metrics = pd.concat(metrics_frames, ignore_index=True)
        rewards = pd.concat(rewards_frames, ignore_index=True)

        # Create one summary row per task
        metrics = metrics.groupby("task", sort=False, as_index=False).mean(
            numeric_only=True
        )
        rewards = rewards.groupby("task", sort=False, as_index=False).mean(
            numeric_only=True
        )

    # Generate figures
    rewards = _plot_rewards(mode=mode, df=rewards, path=figs_dir, window=window)
    performances = _plot_performances(
        mode=mode, df=metrics, path=figs_dir, window=window
    )
    velocities = _plot_velocities(mode=mode, df=metrics, path=figs_dir, window=window)

    return rewards, performances, velocities


def _plot_performances(
    mode: Literal["train", "validation", "evaluation"],
    df: pd.DataFrame,
    path: Path,
    window: int | None,
) -> Figure:
    """
    Plot success, collision, timeout, and travel-time metrics.

    Training metrics are represented as smoothed curves over episodes.
    Validation and evaluation metrics are represented as bars per task.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Create the subplot grid (one for rates and one for travel time)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    df = df.copy()

    # Compute the timeout rate
    df["timeout_rate"] = (1 - df["success_rate"] - df["collision_rate"]).clip(0, 1)

    # Convert rates to percentage
    cols = ["success_rate", "collision_rate", "timeout_rate"]
    df[cols] = 100 * df[cols]

    if mode == "train":
        # Smooth the curves
        cols = [*cols, "mean_time_travel"]
        for col in cols:
            df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

        # Plot
        ax1.plot(df["episode"], df["success_rate_smooth"], linewidth=2, label="Success")
        ax1.plot(
            df["episode"], df["collision_rate_smooth"], linewidth=2, label="Collision"
        )
        ax1.plot(df["episode"], df["timeout_rate_smooth"], linewidth=2, label="Timeout")
        ax2.plot(
            df["episode"],
            df["mean_time_travel_smooth"],
            linewidth=2,
            label="Mean travel time",
        )

        # Configure the figure
        ax1.set_title("Training performance")
        ax2.set_xlabel("Episode")

    else:
        # Sort x labels
        df = df.sort_values(
            "task", key=lambda x: x.str.extract(r"(\d+)$", expand=False).astype(int)
        )

        # Define bars
        x = np.arange(len(df))
        width = 0.25

        # Plot
        ax1.bar(x - width, df["success_rate"], width, label="Success")
        ax1.bar(x, df["collision_rate"], width, label="Collision")
        ax1.bar(x + width, df["timeout_rate"], width, label="Timeout")

        ax2.bar(x, df["mean_time_travel"], label="Mean travel time")

        # Configure the figure
        ax1.set_title(f"{mode.capitalize()} performance")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["task"], rotation=45, ha="right")

    # Configure the figure
    ax1.set_ylabel("Rate (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2.set_ylabel("Travel time")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()

    # Save the figure
    fig.savefig(path / "performances.png", dpi=300)

    return fig


def _plot_velocities(
    mode: Literal["train", "validation", "evaluation"],
    df: pd.DataFrame,
    path: str | Path,
    window: int | None,
) -> Figure:
    """
    Plot mean linear and absolute angular velocities.

    Training metrics are represented as smoothed curves over episodes.
    Validation and evaluation metrics are represented as bars per task.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 4))

    df = df.copy()

    if mode == "train":
        # Smooth the curves
        cols = ["mean_v", "mean_abs_omega"]
        for col in cols:
            df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

        # Plot
        ax.plot(
            df["episode"],
            df["mean_v_smooth"],
            linewidth=1,
            label="Mean v",
            color="tab:blue",
        )

        ax.plot(
            df["episode"],
            df["mean_abs_omega_smooth"],
            linewidth=1,
            label="Mean |omega|",
            color="tab:orange",
        )

        # Configure the figure
        ax.set_xlabel("Episode")

    else:
        # Sort x labels
        df = df.sort_values(
            "task", key=lambda x: x.str.extract(r"(\d+)$", expand=False).astype(int)
        )

        # Define bars
        x = np.arange(len(df))
        width = 0.35

        # Plot
        ax.bar(
            x - width / 2, df["mean_v"], width, label="Mean v (m/s)", color="tab:blue"
        )
        ax.bar(
            x + width / 2,
            df["mean_abs_omega"],
            width,
            label="Mean |omega| (rad/s)",
            color="tab:orange",
        )

        # Configure the figure
        ax.set_xticks(x)
        ax.set_xticklabels(df["task"], rotation=45, ha="right")

    ax.set_ylabel("Velocity")
    ax.set_title("Linear and angular velocities")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    # Save the figure
    fig.savefig(path / "velocities.png", dpi=300)

    return fig


def _plot_rewards(
    mode: Literal["train", "validation", "evaluation"],
    df: pd.DataFrame,
    path: str | Path,
    window: int | None,
) -> Figure:
    """
    Plot returns and reward components.

    Training rewards are represented as smoothed curves over episodes.
    Validation and evaluation returns are represented as bars per task.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 4))

    df = df.copy()

    if mode == "train":
        # Smooth the curves
        cols = [
            "return_total",
            "reward_progress",
            "reward_collision",
            "reward_safety",
            "reward_rotation",
        ]
        for col in cols:
            df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()

        # Plot
        ax.plot(
            df["episode"], df["reward_progress_smooth"], alpha=0.3, label="Progress"
        )
        ax.plot(
            df["episode"], df["reward_collision_smooth"], alpha=0.3, label="Collision"
        )
        ax.plot(df["episode"], df["reward_safety_smooth"], alpha=0.3, label="Safety")
        ax.plot(
            df["episode"], df["reward_rotation_smooth"], alpha=0.3, label="Rotation"
        )
        ax.plot(df["episode"], df["return_total_smooth"], color="black", label="Total")

        # Configure the figure
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward decomposition")

    if mode == "evaluation" or mode == "validation":
        # Sort x labels
        df = df.sort_values(
            "task", key=lambda x: x.str.extract(r"(\d+)$", expand=False).astype(int)
        )

        # Define bars
        x = np.arange(len(df))

        ax.bar(x, df["return_total"], color="tab:blue")

        # Configure the figure
        ax.set_xticks(x)
        ax.set_xticklabels(df["task"], rotation=45, ha="right")

        ax.set_xlabel("Task")
        ax.set_ylabel("Mean return")
        ax.set_title(f"{mode.capitalize()} mean returns")

    # Configure the figure
    ax.grid(axis="y", alpha=0.3)
    ax.legend() if mode == "train" else None

    fig.tight_layout()

    # Save the figure
    fig.savefig(path / "rewards.png", dpi=300)

    return fig
