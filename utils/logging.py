import os

import pandas as pd


def log_train(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:
    """
    Save the training performance metrics to a CSV file and return them as a DataFrame.
    """

    # Create the output directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Convert the metrics to a df and select the columns to save
    df = pd.DataFrame(metrics)
    columns = [
        "experiment",
        "episode",
        "return_total",
        "mean_v",
        "mean_abs_omega",
        "success_rate",
        "collision_rate",
        "mean_time_travel",
    ]
    df = df[[col for col in columns if col in df.columns]]

    # Save to a CSV file
    df.to_csv(f"{path}/{file_name}_training_metrics.csv", index=False)

    return df


def log_debug(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:
    """
    Save the debug information collected during execution to a CSV file and return it as
    a DataFrame.
    """

    # Create the output directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Convert the metrics to a df
    df = pd.DataFrame(metrics)

    # Save to a CSV file
    df.to_csv(f"{path}/{file_name}_debug.csv", index=False)

    return df


def log_rewards(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:
    """
    Save the reward components recorded during training to a CSV file and return them as
    a DataFrame.
    """

    # Create the output directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Convert the metrics to a df and select the columns to save
    df = pd.DataFrame(metrics)
    columns = [
        "experiment",
        "episode",
        "return_total",
        "reward_progress",
        "reward_collision",
        "reward_safety",
        "reward_rotation",
    ]
    df = df[[col for col in columns if col in df.columns]]

    # Save to a CSV file
    df.to_csv(f"{path}/{file_name}_reward_components.csv", index=False)

    return df
