import numpy as np

from configs.config import RewardConfig
from simulator.entities.agent import Agent


def compute_rewards(
    reward_config: RewardConfig, agents: list[Agent], closest: dict, timeout: bool
) -> tuple[dict, dict]:
    """
    Compute the rewards associated with all agents and their components.
    """

    beta1 = reward_config.beta1
    beta2 = reward_config.beta2
    beta3 = reward_config.beta3
    beta4 = reward_config.beta4

    goal_bonus = reward_config.goal_bonus
    collision_malus = reward_config.collision_malus
    angular_malus = reward_config.angular_malus
    safety_malus1 = reward_config.safety_malus1
    safety_malus2 = reward_config.safety_malus2
    omega_threshold = reward_config.omega_threshold
    safety_threshold = reward_config.safety_threshold

    rewards = {}
    rewards_info = {}

    for agent in agents:

        reward_progress = 0.0
        reward_rotation = 0.0
        reward_safety = 0.0
        reward_collision = 0.0

        if agent.state not in ["truncated", "terminated"]:

            current = agent._goal_relative_distance
            progress = agent._old_goal_relative_distance - current

            reward_progress = beta1 * progress
            if current < 0.5:  # target reached
                reward_progress += goal_bonus

            omega = abs(agent.omega)
            if omega > omega_threshold:
                reward_rotation = beta2 * angular_malus * omega

            closest_dist = safety_threshold - closest[agent.id]["closest_distance"]
            if closest_dist > 0:
                reward_safety = (
                    beta3 * safety_malus1 * np.exp(safety_malus2 * closest_dist)
                )

            if agent.state == "collided":
                reward_collision = beta4 * collision_malus

        rewards[agent.id] = (
            reward_progress + reward_collision + reward_safety + reward_rotation
        )
        rewards_info[agent.id] = {
            "reward_progress": reward_progress,
            "reward_collision": reward_collision,
            "reward_safety": reward_safety,
            "reward_rotation": reward_rotation,
        }

    return rewards, rewards_info
