from typing import Literal

import pandas as pd

from configs.config import Task


def get_collision_types(
    policy_name: str,
    checkpoint_name: str,
    tasks: list[Task],
    mode: Literal["validation", "evaluation"],
) -> pd.DataFrame:
    """
    Compute the distribution of collision types for each task.
    """

    collisions = pd.DataFrame(
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
        df = pd.read_csv(
            f"logs/{mode}/{policy_name}/{checkpoint_name}/{task_name}_debug.csv"
        )

        collisions_raw = (
            df.loc[
                df["state"] == "truncated",
                ["task", "episode", "step", "agent", "closest_entity"],
            ]
            .sort_values(["task", "episode", "agent", "step"])
            .drop_duplicates(subset=["task", "episode", "agent"], keep="first")
            .reset_index(drop=True)
        )

        collisions_raw["closest_entity"] = collisions_raw["closest_entity"].replace(
            {"None": pd.NA, "": pd.NA}
        )
        collisions_raw = collisions_raw.dropna(subset=["closest_entity"])

        collisions_raw["collision_type"] = collisions_raw["closest_entity"].str.extract(
            r"^(.+?)_\d+$", expand=False
        )

        row_raw = (
            collisions_raw["collision_type"]
            .value_counts(normalize=True)
            .mul(100)
            .reindex(["agent", "static_entity", "moving_obstacle"], fill_value=0)
            .round(2)
            .T
        )

        collisions.loc[len(collisions)] = {
            "task": task_name,
            "n_collisions": len(collisions_raw),
            "collision_with_agent": row_raw["agent"],
            "collision_with_static": row_raw["static_entity"],
            "collision_with_moving": row_raw["moving_obstacle"],
        }

    return collisions
