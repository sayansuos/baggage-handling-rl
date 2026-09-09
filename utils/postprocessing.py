import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import Task


def get_summary(
    policy_name: str,
    checkpoint_name: str,
    tasks: list[Task],
    mode: Literal["validation", "evaluation"],
    logs_dir: str = "logs",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 4321,
) -> pd.DataFrame:
    """
    Compute task-level metrics, confidence intervals, collision types,
    traveled distances, and speed variations.
    """

    logs_dir = Path(logs_dir) / mode / policy_name / checkpoint_name

    metrics_names = [
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_return",
        "mean_v",
        "mean_abs_omega",
        "mean_travel_time",
    ]

    rows = []

    for i, task in enumerate(tasks):
        task_name = task.name

        metrics_path = logs_dir / f"{task_name}_metrics.csv"
        debug_path = logs_dir / f"{task_name}_debug.csv"

        # Load episode metrics
        metrics = pd.read_csv(metrics_path)
        metrics = metrics.rename(
            columns={
                "mean_time_travel": "mean_travel_time",
                "return_total": "mean_return",
            }
        )
        metrics["timeout_rate"] = (
            1 - metrics["success_rate"] - metrics["collision_rate"]
        ).clip(0, 1)

        row = {"task": task_name, "n_agents": task.env_config.nb_agents}

        # Compute means and bootstrap confidence intervals
        for j, metric in enumerate(metrics_names):
            values = metrics[metric].dropna().to_numpy(dtype=float)

            mean = values.mean()

            lower, upper = _get_bootstrap_ci(
                values=values,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                seed=(seed + i * len(metrics) + j),
            )

            row[metric] = mean
            row[f"{metric}_low"] = lower
            row[f"{metric}_high"] = upper

        # Load debug data
        debug_columns = [
            "task",
            "episode",
            "step",
            "agent",
            "pos_x",
            "pos_y",
            "v",
            "omega",
            "closest_entity",
            "state",
        ]
        debug = pd.read_csv(debug_path, usecols=debug_columns)
        debug = debug.sort_values(["task", "episode", "agent", "step"])

        # Compute collision distributions
        collision_types = _get_collision_types(debug=debug)
        row.update(collision_types)

        # Compute step-level distance and speed variations
        dist_speed = _get_dist_speed(debug=debug)
        row.update(dist_speed)

        # Add information
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = (
        summary.set_index("task").reindex([task.name for task in tasks]).reset_index()
    )

    # Save the processed summary
    summary_path = logs_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    return summary


def _get_bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 1234,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the mean.
    """

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence

    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1.0 - alpha / 2)

    return lower, upper


def _get_collision_types(debug: pd.DataFrame) -> dict:
    """
    Compute the distribution of collision types.
    """

    # Select variables and sort
    collisions = (
        debug.loc[
            debug["state"].eq("truncated"),
            ["task", "episode", "step", "agent", "closest_entity"],
        ]
        .drop_duplicates(subset=["task", "episode", "agent"], keep="first")
        .copy()
    )

    # Replace NA values
    collisions["closest_entity"] = collisions["closest_entity"].replace(
        {"None": pd.NA, "": pd.NA}
    )
    collisions = collisions.dropna(subset=["closest_entity"])

    # Identify collision type
    collisions["collision_type"] = (
        collisions["closest_entity"].astype(str).str.replace(r"_\d+$", "", regex=True)
    )

    # Compute percentages
    collision_percentages = (
        collisions["collision_type"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(["agent", "static_entity", "moving_obstacle"], fill_value=0)
    )

    row = {
        "n_collisions": len(collisions),
        "collision_with_agent": (collision_percentages["agent"]),
        "collision_with_static": (collision_percentages["static_entity"]),
        "collision_with_moving": (collision_percentages["moving_obstacle"]),
    }

    return row


def _get_dist_speed(debug: pd.DataFrame) -> dict:
    """
    Compute the mean traveled distance and the speed variations.
    """

    # Keep data until the first terminal state of each agent
    group_columns = ["task", "episode", "agent"]
    terminal_state = debug["state"].isin(["terminated", "truncated"])
    first_terminal_step = (
        debug["step"]
        .where(terminal_state)
        .groupby([debug["task"], debug["episode"], debug["agent"]])
        .transform("min")
    )
    motion = debug.loc[
        first_terminal_step.isna() | debug["step"].le(first_terminal_step)
    ].copy()

    # Compute distance between steps
    delta_x = motion.groupby(group_columns)["pos_x"].diff()
    delta_y = motion.groupby(group_columns)["pos_y"].diff()
    motion["step_distance"] = np.hypot(delta_x, delta_y)

    # Compute speed variation between steps
    motion["abs_delta_v"] = motion.groupby(group_columns)["v"].diff().abs()
    motion["abs_delta_omega"] = motion.groupby(group_columns)["omega"].diff().abs()

    # Aggregate per agent and episode
    agent_episodes = motion.groupby(group_columns, as_index=False, sort=False).agg(
        distance_traveled=("step_distance", "sum"),
        mean_abs_delta_v=("abs_delta_v", "mean"),
        mean_abs_delta_omega=("abs_delta_omega", "mean"),
    )

    row = {
        "mean_distance_traveled": (agent_episodes["distance_traveled"].mean()),
        "mean_abs_delta_v": (agent_episodes["mean_abs_delta_v"].mean()),
        "mean_abs_delta_omega": (agent_episodes["mean_abs_delta_omega"].mean()),
    }

    return row


def plot_training_heatmap(
    policy_name: str,
    metric: Literal[
        "sucess_rate",
        "collision_rate",
        "timeout_rate",
        "mean_v",
        "mean_abs_omega",
        "return_total",
    ],
    tasks: list[Task],
    path: str | Path,
    cmap: str = "RdYlGn",
    n_bins: int = 50,
) -> pd.DataFrame:
    """
    Plot the evolution of a training metric for each curriculum task.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Cut the training into n_bins periods
    df = pd.read_csv(f"logs/train/{policy_name}/metrics.csv")
    df = df.sort_values(
        "task", key=lambda x: x.str.extract(r"(\d+)$", expand=False).astype(int)
    )
    df["training_bin"] = pd.cut(df["episode"], bins=n_bins, labels=False)

    if metric == "timeout_rate":
        df["timeout_rate"] = df.apply(
            lambda x: max(0, 1 - x["success_rate"] - x["collision_rate"]), axis=1
        )

    # Average metric rate per task and period
    values = df.groupby(["task", "training_bin"])[metric].mean().unstack()

    # Plot heatmap
    task_names = [task.name for task in tasks]
    values = values.reindex(index=task_names, columns=range(n_bins))
    task_labels = [_get_title(task_name=task_name) for task_name in task_names]
    if metric == "return_total":
        vmin, vmax = np.floor(values.min().min()), np.ceil(values.max().max())
    elif metric == "mean_v":
        vmin, vmax = 0.0, 2.0
    elif metric == "mean_abs_omega":
        vmin, vmax = 0.0, np.pi / 3
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(
        values, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap
    )
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_labels)

    # Display % progression
    ticks = np.linspace(0, n_bins - 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(x)} %" for x in np.linspace(0, 100, 6)])

    ax.set_xlabel("Progression in the training")
    ax.set_ylabel("Task")
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(f"Average {metric.replace('_', ' ')}")

    plt.tight_layout()
    fig.savefig(path / f"{metric}.png", dpi=300)

    return values


def _get_title(task_name: str) -> str:
    """
    Return the str use for naming the task.
    """
    match = re.fullmatch(r"(task|eval)_?(\d+)", task_name)

    if match is None:
        return task_name

    section, task_id = match.groups()
    prefix = "T" if section == "task" else "E"

    return rf"${prefix}_{{{task_id}}}$"
