import os

import pandas as pd


def log_train(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:

    os.makedirs(path, exist_ok=True)
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
    df.to_csv(f"{path}/{file_name}_training_metrics.csv", index=False)

    return df


def log_debug(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:

    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(f"{path}/{file_name}_debug.csv", index=False)

    return df


def log_rewards(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:

    os.makedirs(path, exist_ok=True)

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

    df.to_csv(f"{path}/{file_name}_reward_components.csv", index=False)

    return df
