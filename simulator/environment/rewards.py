import numpy as np

from configs.config import RewardConfig
from simulator.entities.agent import Agent


def compute_rewards(
    reward_config: RewardConfig, agents: list[Agent], closest: dict, timeout: bool
) -> dict[str, float]:
    """
    Compute the rewards associated with all agents.
    """

    rewards = {}

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

    for agent in agents:

        if agent.state in ["truncated", "terminated"]:
            rewards[agent.id] = 0.0

        else:
            reward = 0.0

            current = agent._goal_relative_distance
            progress = agent._old_goal_relative_distance - current

            # Progress reward
            reward += beta1 * progress

            # Goal reached
            if current < 0.5:
                reward += goal_bonus

            # Abrupt rotations penalty
            omega = abs(agent.omega)
            if omega > omega_threshold:
                reward += beta2 * angular_malus * omega

            # Non-respect of safety distance penalty
            closest_dist = safety_threshold - closest[agent.id]["closest_distance"]
            if closest_dist > 0:
                reward += beta3 * safety_malus1 * np.exp(safety_malus2 * closest_dist)

            # Collision penalty
            if agent.state == "collided":
                reward += beta4 * collision_malus

            rewards[agent.id] = reward

    return rewards
