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

    margin: int = 5
    max_attempts: int = 100
    max_steps: int = 300

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
    length_view: int = 15
    length_view_decrease: int | None = None

    collision_threshold: float = 0.5
    reach_threshold: float = 0.5


@dataclass
class RewardConfig:
    beta1: float = 5.0
    beta2: float = 5.1
    beta3: float = 1.0

    linear_bonus_factor: float = 2.0
    progress_reduction_factor: float = 0.2
    angular_malus_factor: float = -2.0
    safety_malus_factor: float = -1.5

    goal_bonus: float = 400.0
    step_malus: float = -2.0
    collision_malus: float = -200.0

    safety_threshold: float = 3.0


@dataclass
class Task:
    name: str
    env_config: EnvConfig
    agent_config: AgentConfig
    reward_config: RewardConfig
    n_steps: int | None


@dataclass
class SACConfig:
    obs_size: int = 7
    tau: float = 0.005
    alpha: float = 0.15
    batch_size: int = 64
    critic_lr: float = 3e-4
    actor_lr: float = 3e-5
    gamma: float = 0.99
    reparam_noise: float = 1e-6
    feature_size: int = 64
    hidden_size: int = 128
    mem_size: int = 100_000


@dataclass
class Curriculum:
    chunk_steps: int = 10_000
    n_chunks: int = 5
    threshold: float = 0.9
    n_eval_episodes: int = 100
