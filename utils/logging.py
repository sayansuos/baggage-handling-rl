from pathlib import Path
from typing import Literal

import pandas as pd


def log(
    history: list[dict],
    debug: list[dict] | None,
    logs_dir: str,
    mode: Literal["train", "validation", "evaluation"],
    policy_name: str,
    checkpoint_name: str | None,
    file_name: str | None,
) -> None:
    """
    Save all metrics to a CSV file.
    The output files are saved in logs_dir/mode/policy_name/checkpoint_name/.
    """
    if mode == "train":
        path = Path(logs_dir) / mode / policy_name
    else:
        path = Path(logs_dir) / mode / policy_name / checkpoint_name

    log_metrics(metrics=history, path=path, file_name=file_name)
    log_rewards(metrics=history, path=path, file_name=file_name)
    if debug is not None:
        log_debug(metrics=debug, path=path, file_name=file_name)


def log_metrics(metrics: list[dict], path: Path, file_name: str | None) -> None:
    """
    Save performance metrics to a CSV file.
    """

    # Create the output directory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    # Convert the metrics to a df and select the columns to save
    columns = [
        "task",
        "episode",
        "return_total",
        "mean_v",
        "mean_abs_omega",
        "success_rate",
        "collision_rate",
        "mean_time_travel",
    ]
    df = pd.DataFrame(metrics).reindex(columns=columns)

    # Save to a CSV file
    output_file = "metrics.csv" if file_name is None else f"{file_name}_metrics.csv"
    df.to_csv(path / output_file, index=False)


def log_debug(metrics: list[dict], path: Path, file_name: str | None) -> None:
    """
    Save detailed debug information to a CSV file.
    """

    # Create the output directory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    # Convert the metrics to a df
    df = pd.DataFrame(metrics)

    # Save to a CSV file
    output_file = "debug.csv" if file_name is None else f"{file_name}_debug.csv"
    df.to_csv(path / output_file, index=False)


def log_rewards(metrics: list[dict], path: Path, file_name: str | None) -> None:
    """
    Save reward components to a CSV file.
    """

    # Create the output directory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    # Convert the metrics to a df and select the columns to save
    columns = [
        "task",
        "episode",
        "return_total",
        "reward_progress",
        "reward_collision",
        "reward_safety",
        "reward_rotation",
    ]
    df = pd.DataFrame(metrics).reindex(columns=columns)

    # Save to a CSV file
    output_file = "rewards.csv" if file_name is None else f"{file_name}_rewards.csv"
    df.to_csv(path / output_file, index=False)
