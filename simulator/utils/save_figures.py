import cv2
import pandas as pd
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt


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

    imageio.imwrite(f"{path}/{file_name}.png", img)


def save_rewards(
    df: pd.DataFrame,
    file_name: str,
    path: str,
):
    """
    Plot reward evolution for multiple episodes in a single figure.
    """

    experiments = df["environment"].unique()
    fig, axes = plt.subplots(
        nrows=len(experiments),
        ncols=2,
        figsize=(12, 4 * len(experiments)),
        squeeze=False,
    )

    for row, exp in enumerate(experiments):

        data = df[df["environment"] == exp].copy()

        success_steps = (
            data.loc[data["state"] == "terminated"]
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
            r = data.loc[data["agent"] == agent, "reward"]
            ax_reward.plot(steps, r, alpha=0.3, label=agent, linestyle="--")
            ax_cumsum.plot(steps, r.cumsum(), alpha=0.3, label=agent, linestyle="--")

        ax_reward.plot(steps, rewards, color="black")
        for s in collision_steps:
            ax_reward.axvline(s, color="red", alpha=0.7)
        for s in success_steps:
            ax_reward.axvline(s, color="green", alpha=0.7)
        ax_reward.set_title(f"REWARD - {exp}")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.legend()
        ax_reward.grid(True)

        ax_cumsum.plot(steps, rewards.cumsum(), color="black")
        for s in collision_steps:
            ax_cumsum.axvline(s, color="red", alpha=0.7)
        for s in success_steps:
            ax_cumsum.axvline(s, color="green", alpha=0.7)

        ax_cumsum.set_title(f"CUMULATIVE REWARD - {exp}")
        ax_cumsum.set_xlabel("Step")
        ax_cumsum.legend()
        ax_cumsum.grid(True)

    fig.tight_layout()
    fig.savefig(f"{path}/{file_name}.png", dpi=300)
    plt.close(fig)
