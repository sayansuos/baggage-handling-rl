import os

import pandas as pd


def log_training_metrics(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:

    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(f"{path}/{file_name}_global_metrics.csv", index=False)

    return df


def log_metrics(
    metrics: list[dict],
    path: str,
    file_name: str = "",
) -> pd.DataFrame:
    """
    Save SAC training metrics and plot return evolution.
    """

    os.makedirs(path, exist_ok=True)

    rows = []

    for ep in metrics:
        rows.append(
            {
                "environment": ep["environment"],
                "worker": ep["worker"],
                "episode": ep["episode"],
                "return": ep["return"],
                "mean_time_travel": ep["mean_time_travel"],
                "success_rate": ep["success_rate"],
                "collision_rate": ep["collision_rate"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{path}/{file_name}_metrics.csv", index=False)

    return df


def log_metrics_debug(
    metrics: list[dict], path: str, file_name: str = ""
) -> pd.DataFrame:

    os.makedirs(path, exist_ok=True)

    rows = []

    for ep in metrics:

        debug_steps = ep.get("debug", [])

        for step in debug_steps:
            agents = step.get("agents", {})

            for agent_id, agent_data in agents.items():
                rows.append(
                    {
                        "environment": ep.get("environment"),
                        "worker": ep["worker"],
                        "episode": ep["episode"],
                        "step": step["step"],
                        "agent": agent_id,
                        # "old_pos": agent_data.get("old_pos"),
                        # "current_pos": agent_data.get("current_pos"),
                        "state": agent_data["state"],
                        "reward": agent_data["reward"],
                        "goal_distance": agent_data["goal_relative_distance"],
                        "travel_time": agent_data["travel_time"],
                        "action_x": agent_data.get("action", [0, 0])[0],
                        "action_y": agent_data.get("action", [0, 0])[1],
                        "closest_obstacle_distance": agent_data[
                            "closest_obstacle_distance"
                        ],
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(f"{path}/{file_name}_debug.csv", index=False)

    return df
