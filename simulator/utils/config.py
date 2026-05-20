from dataclasses import dataclass
import numpy as np


@dataclass
class EnvConfig:
    width: int = 100
    height: int = 60

    agent_mode: str = "random"
    env_mode: str = "random"

    nb_agents: int = 1
    nb_static_obstacles: int = 20
    nb_moving_obstacles: int = 3
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
    radius_max: float = 2
    margin: int = 4
    max_attempts: int = 100


@dataclass
class AgentConfig:
    v_min: float = 0.0
    v_max: float = 5.0
    omega_min: float = -np.pi / 6
    omega_max: float = np.pi / 6

    radius: float = 0.5
    length_view: int = 5
