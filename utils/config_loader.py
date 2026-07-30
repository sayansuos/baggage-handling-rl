import yaml

from configs.config import AgentConfig, EnvConfig, RewardConfig, Task


def load_tasks(task_path="configs/tasks.yaml", obj: str = "train") -> list[Task]:
    """
    Load a list of task configurations from a YAML file.
    """

    tasks = []

    # Open the configuration file
    with open(task_path, "r") as f:
        data = yaml.safe_load(f)

    # Select the task section according to the requested mode
    for task in data[obj]:
        # Load the configurations according to the task
        env_config = load_env_config(name=task["env_config"], obj=obj)
        agent_config = load_agent_config(name=task["agent_config"])
        reward_config = load_reward_config(name=task["reward_config"])

        # Build the task object
        tasks.append(
            Task(
                name=task["name"],
                env_config=env_config,
                agent_config=agent_config,
                reward_config=reward_config,
                n_steps=task["n_steps"],
            )
        )

    return tasks


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
    configs = data[obj]

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
