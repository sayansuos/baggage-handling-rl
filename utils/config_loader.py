import yaml

from configs.config import AgentConfig, EnvConfig, Experiment, RewardConfig


def load_experiments(
    exp_path="configs/experiments.yaml", obj: str = "train"
) -> list[Experiment]:
    """
    Load a list of experiment configurations from a YAML file.
    """

    experiments = []

    # Open the configuration file
    with open(exp_path, "r") as f:
        data = yaml.safe_load(f)

    # Select the experiment section according to the requested mode
    for exp in data[f"{obj}_experiments"]:
        # Load the configurations according to the experiment
        env_config = load_env_config(name=exp["env_config"], obj=obj)
        agent_config = load_agent_config(name=exp["agent_config"])
        reward_config = load_reward_config(name=exp["reward_config"])

        # Build the experiment object
        experiments.append(
            Experiment(
                name=exp["name"],
                env_config=env_config,
                agent_config=agent_config,
                reward_config=reward_config,
                n_steps=exp["n_steps"],
            )
        )

    return experiments


def load_env_config(name: str | None = None, obj: str = "train") -> EnvConfig:
    """
    Load environment configuration from a YAML file.
    """

    # Return the default configuration when no name is provided
    if not name:
        return EnvConfig()

    # Open the configuration file
    with open("configs/environments.yaml", "r") as f:
        data = yaml.safe_load(f)

    # Select the environment section according to the requested mode
    configs = data[f"{obj}_environments"]

    # Find the configuration whose name matches the requested name and copy
    conf = next(conf for conf in configs if conf["name"] == name)
    conf = conf.copy()

    # Remove the name field
    conf.pop("name")

    return EnvConfig(**conf)


def load_agent_config(name: str | None = None) -> AgentConfig:
    """
    Load agent configuration from a YAML file.
    """

    # Return the default configuration when no name is provided
    if not name:
        return AgentConfig()

    # Open the configuration file
    with open("configs/agents.yaml", "r") as f:
        data = yaml.safe_load(f)

    # Find the configuration whose name matches the requested name and copy
    configs = data["agents"]
    conf = next(conf for conf in configs if conf["name"] == name)
    conf = conf.copy()

    # Remove the name field
    conf.pop("name")

    return AgentConfig(**conf)


def load_reward_config(name: str | None = None) -> RewardConfig:
    """
    Load reward configuration from a YAML file.
    """

    # Return the default configuration when no name is provided
    if not name:
        return RewardConfig()

    # Open the configuration file
    with open("configs/rewards.yaml", "r") as f:
        data = yaml.safe_load(f)

    # Find the configuration whose name matches the requested name and copy
    configs = data["rewards"]
    conf = next(conf for conf in configs if conf["name"] == name)
    conf = conf.copy()

    # Remove the name field
    conf.pop("name")

    return RewardConfig(**conf)
