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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the metrics summary for analysis.
    """

    summary = pd.DataFrame(
        columns=[
            "task",
            "n_agents",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_travel_time",
            "mean_return",
        ]
    )

    metrics = [
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_travel_time",
        "mean_return",
    ]
    ic = []

    for task in tasks:
        task_name = task.name
        df = pd.read_csv(
            f"logs/{mode}/{policy_name}/{checkpoint_name}/{task_name}_metrics.csv"
        )
        df = df.rename(
            columns={
                "mean_time_travel": "mean_travel_time",
                "return_total": "mean_return",
            }
        )
        df["timeout_rate"] = df.apply(
            lambda x: max(0, 1 - x["success_rate"] - x["collision_rate"]), axis=1
        )
        row = {
            "task": task_name,
            "n_agents": task.env_config.nb_agents,
            "success_rate": df["success_rate"].mean().round(2),
            "collision_rate": df["collision_rate"].mean().round(2),
            "timeout_rate": df["timeout_rate"].mean().round(2),
            "mean_travel_time": df["mean_travel_time"].mean().round(2),
            "mean_return": df["mean_return"].mean().astype(int),
        }
        summary.loc[len(summary)] = row

        ic_row = {"task": task.name}
        for metric in metrics:
            values = df[metric].to_numpy()
            lower, upper = get_bootstrap_ci(
                values=values, confidence=0.95, n_bootstrap=10_000, seed=1234
            )
            ic_row[f"{metric}_low"] = round(lower, 2)
            ic_row[f"{metric}_high"] = round(upper, 2)

        ic.append(ic_row)

    ic = pd.DataFrame(ic)

    return summary, ic


def get_bootstrap_ci(
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


def get_collision_types(
    policy_name: str,
    checkpoint_name: str,
    tasks: list[Task],
    mode: Literal["validation", "evaluation"],
) -> pd.DataFrame:
    """
    Compute the distribution of collision types for each task.
    """

    df = pd.DataFrame(
        columns=[
            "task",
            "n_collisions",
            "collision_with_agent",
            "collision_with_static",
            "collision_with_moving",
        ]
    )

    for task in tasks:
        task_name = task.name
        debug = pd.read_csv(
            f"logs/{mode}/{policy_name}/{checkpoint_name}/{task_name}_debug.csv"
        )

        df_raw = (
            debug.loc[
                debug["state"] == "truncated",
                ["task", "episode", "step", "agent", "closest_entity"],
            ]
            .sort_values(["task", "episode", "agent", "step"])
            .drop_duplicates(subset=["task", "episode", "agent"], keep="first")
            .reset_index(drop=True)
        )

        df_raw["closest_entity"] = df_raw["closest_entity"].replace(
            {"None": pd.NA, "": pd.NA}
        )
        df_raw = df_raw.dropna(subset=["closest_entity"])

        df_raw["collision_type"] = df_raw["closest_entity"].str.extract(
            r"^(.+?)_\d+$", expand=False
        )

        row_raw = (
            df_raw["collision_type"]
            .value_counts(normalize=True)
            .mul(100)
            .reindex(["agent", "static_entity", "moving_obstacle"], fill_value=0)
            .round(2)
            .T
        )

        df.loc[len(df)] = {
            "task": task_name,
            "n_collisions": len(df_raw),
            "collision_with_agent": row_raw["agent"],
            "collision_with_static": row_raw["static_entity"],
            "collision_with_moving": row_raw["moving_obstacle"],
        }

    return df


def get_dist_speed(
    policy_name: str,
    checkpoint_name: str,
    tasks: list[Task],
    mode: Literal["validation", "evaluation"],
) -> pd.DataFrame:
    """
    Compute the mean traveled distance and the speed variations for each task.
    """

    columns = ["task", "episode", "step", "agent", "pos_x", "pos_y", "v", "omega"]

    dataframes = []
    for task in tasks:
        task_name = task.name
        df = pd.read_csv(
            f"logs/{mode}/{policy_name}/{checkpoint_name}/{task_name}_debug.csv"
        )
        dataframes.append(df[columns])
    debug = pd.concat(dataframes, ignore_index=True)

    group_columns = ["task", "episode", "agent"]
    debug = debug.sort_values([*group_columns, "step"])

    # Compute distance between steps
    delta_x = debug.groupby(group_columns)["pos_x"].diff()
    delta_y = debug.groupby(group_columns)["pos_y"].diff()
    debug["step_distance"] = np.hypot(delta_x, delta_y)

    # Compute speed variation between steps
    debug["abs_delta_v"] = debug.groupby(group_columns)["v"].diff().abs()
    debug["abs_delta_omega"] = debug.groupby(group_columns)["omega"].diff().abs()

    # Aggregate all
    summary = debug.groupby(group_columns, as_index=False, sort=False).agg(
        n_steps=("step", "count"),
        distance_traveled=("step_distance", "sum"),
        mean_v=("v", "mean"),
        mean_abs_omega=("omega", lambda x: x.abs().mean()),
        mean_abs_delta_v=("abs_delta_v", "mean"),
        mean_abs_delta_omega=("abs_delta_omega", "mean"),
    )

    # Compute mean per tasl
    df = (
        summary.groupby("task", as_index=False)
        .agg(
            mean_distance_traveled=("distance_traveled", "mean"),
            mean_abs_delta_v=("mean_abs_delta_v", "mean"),
            mean_abs_delta_omega=("mean_abs_delta_omega", "mean"),
        )
        .round(3)
    )

    return df


def plot_training_heatmap(
    policy_name: str,
    metric: str,
    tasks: list[Task],
    path: str | Path,
    cmap: str = "RdYlGn",
    n_bins: int = 50,
) -> None:
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
    task_labels = (
        df["task"]
        .map(lambda task_name: _get_title(task_name=task_name))
        .unique()
        .tolist()
    )
    values = values.reindex(task_names)
    vmin = np.floor(values.min().min())
    vmax = np.ceil(values.max().max())
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
