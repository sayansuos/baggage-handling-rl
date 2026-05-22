import pandas as pd


def save_as_df(history: list[dict], file_name: str, path="logs/") -> pd.DataFrame:
    rows = []

    for step_data in history:
        episode = step_data["episode"]
        step = step_data["step_count"]
        rewards = step_data["reward"]
        agents = step_data["agents"]
        for agent_name, agent_data in agents.items():
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "agent": agent_name,
                    "state": agent_data["state"],
                    "reward": rewards[agent_name],
                    "goal_distance": agent_data["goal_relative_distance"],
                    "time_travel": agent_data["time_travel"],
                    "action_x": agent_data["action"][0],
                    "action_y": agent_data["action"][1],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(path + file_name, index=False)

    return df
