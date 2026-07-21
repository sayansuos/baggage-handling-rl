from configs.config import RewardConfig
from simulator.entities.agent import Agent


def compute_rewards(
    reward_config: RewardConfig, agents: list[Agent]
) -> tuple[dict, dict]:
    """
    Compute the rewards associated with all agents and their components.
    """

    # Get the reward coefficients
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

    # Initialize the rewards dicts
    rewards = {}
    rewards_info = {}

    for agent in agents:
        # Initialize the reward components
        reward_progress = 0.0
        reward_rotation = 0.0
        reward_safety = 0.0
        reward_collision = 0.0

        # Skip reward computation for inactive agents.
        if agent.state not in ["truncated", "terminated"]:
            current = agent._goal_relative_distance  # Relative distance
            progress = agent._old_goal_relative_distance - current  # Distance progress
            omega = abs(agent.omega)  # Angular velocity

            ### -----------------------------------------------------
            ### Compute the progress reward
            ### -----------------------------------------------------

            reward_progress = beta1 * progress

            # Reward high speed
            reward_progress += linear_bonus_factor * agent.v

            # Reduce progress_reward if near obstacles
            if agent._closest_dist < safety_threshold:
                reward_progress *= progress_reduction_factor

            # Penalize steps
            reward_progress += step_malus

            # Reward target reach
            if agent.state == "reached":
                reward_progress += goal_bonus

            ### -----------------------------------------------------
            ### Compute the rotation penalty
            ### -----------------------------------------------------

            reward_rotation = beta2 * angular_malus_factor * (omega**2)

            ### -----------------------------------------------------
            ### Compute the obstacle penalty
            ### -----------------------------------------------------

            if agent._closest_dist < safety_threshold:
                d = max(agent._closest_dist, 0.1)
                reward_safety = (
                    beta3 * safety_malus_factor * (1 / d**2 - 1 / safety_threshold**2)
                )

            ### -----------------------------------------------------
            ### Compute the collision penalty
            ### -----------------------------------------------------

            if agent.state == "collided":
                reward_collision = collision_malus

        # Store the total reward and its components
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
