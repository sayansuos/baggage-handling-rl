import yaml

from configs.config import AgentConfig, EnvConfig, Experiment, RewardConfig


def load_experiments(path="configs/experiments.yaml") -> list[Experiment]:
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
                env_config=load_env_config(exp["env_config"]),
                agent_config=load_agent_config(exp["agent_config"]),
                reward_config=load_reward_config(exp["reward_config"]),
                n_steps=exp["n_steps"],
            )
        )

    return experiments


def load_env_config(name: str | None = None) -> EnvConfig:
    """
    Load environment configuration from a YAML file.
    """

    if not name:
        return EnvConfig()

    with open("configs/environments.yaml", "r") as f:
        data = yaml.safe_load(f)

    configs = data["environments"]
    conf = next(conf for conf in configs if conf["name"] == name)
    conf = conf.copy()
    conf.pop("name")

    return EnvConfig(**conf)


def load_agent_config(name: str | None = None) -> AgentConfig:
    """
    Load agent configuration from a YAML file.
    """

    if not name:
        return AgentConfig()

    with open("configs/agents.yaml", "r") as f:
        data = yaml.safe_load(f)

    configs = data["agents"]
    conf = next(conf for conf in configs if conf["name"] == name)
    conf = conf.copy()
    conf.pop("name")

    return AgentConfig(**conf)


def load_reward_config(name: str | None = None) -> RewardConfig:
    """
    Load reward configuration from a YAML file.
    """

    if not name:
        return RewardConfig()

    with open("configs/rewards.yaml", "r") as f:
        data = yaml.safe_load(f)

    configs = data["rewards"]
    conf = next(conf for conf in configs if conf["name"] == name)
    conf = conf.copy()
    conf.pop("name")

    return RewardConfig(**conf)
