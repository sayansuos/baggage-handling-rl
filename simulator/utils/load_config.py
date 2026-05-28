import yaml

from simulator.configs.config import EnvConfig, AgentConfig, RewardConfig, Experiment


def load_experiments(path="simulator/configs/experiments.yaml") -> list[Experiment]:
    """
    Load a list of experiment configurations from a YAML file.
    """

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    experiments = []

    for exp in data["experiments"]:
        experiments.append(
            Experiment(
                name=exp["name"],
                env_config=load_env_config(exp["env_config_path"]),
                agent_config=load_agent_config(exp["agent_config_path"]),
                reward_config=load_reward_config(exp["reward_config_path"]),
                n_envs=exp["n_envs"],
                n_episodes=exp["n_episodes"],
            )
        )

    return experiments


def load_env_config(path: str | None = None) -> EnvConfig:
    """
    Load environment configuration from a YAML file.
    """

    if not path:
        return EnvConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return EnvConfig(**data)


def load_agent_config(path: str | None = None) -> AgentConfig:
    """
    Load agent configuration from a YAML file.
    """

    if not path:
        return AgentConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AgentConfig(**data)


def load_reward_config(path: str | None = None) -> RewardConfig:
    """
    Load reward configuration from a YAML file.
    """

    if not path:
        return RewardConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return RewardConfig(**data)
