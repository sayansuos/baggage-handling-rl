import pandas as pd


def save_as_df(history: list[dict], file_name: str, path="logs/"):
    rows_info = []
    rows_debug = []

    for ep in history:
        rows_info.append(
            {
                "environment": ep.get("environment"),
                "episode": ep["episode"],
                "return": ep["return"],
                "mean_time_travel": ep["mean_time_travel"],
                "success_rate": ep["success_rate"],
                "collision_rate": ep["collision_rate"],
            }
        )
        debug_steps = ep.get("debug", [])
        for step in debug_steps:
            agents = step.get("agents", {})
            for agent_id, agent_data in agents.items():
                rows_debug.append(
                    {
                        "environment": ep.get("environment"),
                        "episode": ep["episode"],
                        "step": step["step"],
                        "agent": agent_id,
                        "old_pos": agent_data.get("old_pos"),
                        "current_pos": agent_data.get("current_pos"),
                        "state": agent_data.get("state"),
                        "reward": agent_data.get("reward"),
                        "goal_distance": agent_data.get("goal_relative_distance"),
                        "travel_time": agent_data.get("travel_time"),
                        "action_x": agent_data.get("action", [0, 0])[0],
                        "action_y": agent_data.get("action", [0, 0])[1],
                        "closest_obstacle_distance": agent_data.get(
                            "closest_obstacle_distance"
                        ),
                    }
                )

    df = pd.DataFrame(rows_info)
    df.to_csv(path + file_name + ".csv", index=False)
    df_debug = pd.DataFrame(rows_debug)
    df_debug.to_csv(path + file_name + "_debug.csv", index=False)

    return df
