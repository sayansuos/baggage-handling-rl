import yaml

from simulator.configs.config import EnvConfig, AgentConfig, RewardConfig


def load_env_config(path: str | None = None) -> EnvConfig:

    if not path:
        return EnvConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return EnvConfig(**data)


def load_agent_config(path: str | None = None) -> AgentConfig:

    if not path:
        return AgentConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AgentConfig(**data)


def load_reward_config(path: str | None = None) -> RewardConfig:

    if not path:
        return RewardConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return RewardConfig(**data)
