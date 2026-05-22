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
    nb_moving_obstacles: int = 5
    nb_targets: int = 2

    v_min_allowed: float = 0.0
    v_max_allowed: float = 5.0
    omega_min_allowed: float = -np.pi / 6
    omega_max_allowed: float = np.pi / 6

    width_min: int = 1
    width_max: int = 20
    height_min: int = 1
    height_max: int = 20
    thickness: int = 1
    radius_min: float = 0.5
    radius_max: float = 1
    margin: int = 3
    max_attempts: int = 100


@dataclass
class AgentConfig:
    v_min: float = 0.0
    v_max: float = 5.0
    omega_min: float = -np.pi / 6
    omega_max: float = np.pi / 6

    radius: float = 0.5
    length_view: int = 5


@dataclass
class RewardConfig:
    beta1: float = 5
    beta2: float = 0.1
    beta3: float = 0.1
    beta4: float = 1

    goal_bonus: float = 2
    collision_malus: float = -10
    angular_malus: float = -0.01
    omega_threshold: float = np.pi / 12
    safety_malus1: float = -0.1
    safety_malus2: float = 1
    safety_threshold: float = 1
