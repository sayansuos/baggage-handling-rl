from pathlib import Path

import pandas as pd


def log_metrics(metrics: list[dict], path: str | Path, file_name: str) -> None:
    """
    Save performance metrics to a CSV file.

    The output file is named ''<file_name>_metrics.csv''.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Convert the metrics to a df and select the columns to save
    df = pd.DataFrame(metrics)
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
    df.to_csv(path / f"{file_name}_metrics.csv", index=False)


def log_debug(metrics: list[dict], path: str | Path, file_name: str) -> None:
    """
    Save detailed debug information to a CSV file.

    The output file is named ''<file_name>_debug.csv''.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Convert the metrics to a df
    df = pd.DataFrame(metrics)

    # Save to a CSV file
    df.to_csv(path / f"{file_name}_debug.csv", index=False)


def log_rewards(metrics: list[dict], path: str | Path, file_name: str) -> None:
    """
    Save reward components to a CSV file.

    The output file is named ''<file_name>_rewards.csv''.
    """

    # Create the output directory if it does not exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Convert the metrics to a df and select the columns to save
    df = pd.DataFrame(metrics)
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
    df.to_csv(path / f"{file_name}_rewards.csv", index=False)
