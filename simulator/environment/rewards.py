from configs.config import RewardConfig
from simulator.entities.agent import Agent


def compute_rewards(
    reward_config: RewardConfig, agents: list[Agent]
) -> tuple[dict, dict]:
    """
    Compute the rewards associated with all agents and their components.
    """

    beta1 = reward_config.beta1
    beta2 = reward_config.beta2
    beta3 = reward_config.beta3

    linear_bonus_factor = reward_config.linear_bonus_factor
    progress_reduction_factor = reward_config.progress_reduction_factor
    step_malus = reward_config.step_malus
    goal_bonus = reward_config.goal_bonus
    collision_malus = reward_config.collision_malus
    angular_malus_factor = reward_config.angular_malus_factor
    safety_malus_factor = reward_config.safety_malus_factor
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
            omega = abs(agent.omega)

            # Progress reward
            if agent._closest_dist >= safety_threshold:
                reward_progress = beta1 * progress - step_malus

            else:
                reward_progress = (
                    beta1 * progress_reduction_factor * progress - step_malus
                )
            reward_progress += linear_bonus_factor * agent.v
            if agent.state == "reached":
                reward_progress += goal_bonus

            # Rotation penalty
            reward_rotation = beta2 * angular_malus_factor * (omega**2)

            # Obstacle penalty
            if agent._closest_dist < safety_threshold:
                d = max(agent._closest_dist, 0.1)
                reward_safety = (
                    beta3 * safety_malus_factor * (1 / d**2 - 1 / safety_threshold**2)
                )

            # Collision penalty
            if agent.state == "collided":
                reward_collision = collision_malus

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
