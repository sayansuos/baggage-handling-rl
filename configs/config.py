from dataclasses import dataclass

import numpy as np


@dataclass
class EnvConfig:
    width: int = 96
    height: int = 48

    agent_mode: str = "random"
    env_mode: str = "random"

    nb_agents: int = 1
    nb_static_obstacles: int = 10
    nb_moving_obstacles: int = 0
    nb_targets: int = 2

    margin: int = 4
    max_attempts: int = 100
    max_steps: int = 300

    width_min: int = 1
    width_max: int = 20
    height_min: int = 1
    height_max: int = 20
    thickness: int = 1
    radius_min: float = 0.5
    radius_max: float = 1


@dataclass
class AgentConfig:

    n_maps: int = 3

    v_min: float = 0.0
    v_max: float = 5.0
    omega_min: float = -np.pi / 3
    omega_max: float = np.pi / 3

    radius: float = 0.5
    length_view: int = 11


@dataclass
class RewardConfig:
    beta1: float = 5.0
    beta2: float = 0.1
    beta3: float = 0.5

    goal_bonus: float = 400.0
    collision_malus: float = -100.0
    angular_malus: float = -2.0
    safety_malus: float = -1.0
    safety_threshold: float = 2.0


@dataclass
class Experiment:
    name: str
    env_config: EnvConfig
    agent_config: AgentConfig
    reward_config: RewardConfig
    n_steps: int
