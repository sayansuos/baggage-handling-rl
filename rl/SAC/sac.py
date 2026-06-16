import warnings

import numpy as np
import pandas as pd

from configs.config import Experiment
from rl.sac.agent import SACAgent
from simulator.environment.environment import Environment

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_sac(
    exp: Experiment,
    n_steps: int = 10000,
    warmup_steps: int = 1000,
    update_frequency: int = 4,
    trained_agent_id: str = "agent_1",
):
    """ """

    # Create environment and SAC Agent

    env = Environment(
        exp.env_config,
        exp.agent_config,
        exp.reward_config,
        exp.name,
    )

    agent = SACAgent(
        env_name=f"{exp.name}",
        map_shape=(
            1,
            exp.agent_config.length_view,
            exp.agent_config.length_view,
        ),
        action_space=env.action_space[trained_agent_id],
    )

    history = []
    debug = []
    best_score = -np.inf
    scores = []

    total_steps = 0
    episode = 0

    while total_steps < n_steps:
        state, _ = env.reset()
        episode += 1
        score = 0.0

        if env.episode % 20 == 0:
            env.debug = True
        else:
            env.debug = False

        while not env.dones[trained_agent_id] and total_steps < n_steps:
            actions = {}

            # Warmup to fill the RB with random data
            if total_steps < warmup_steps:
                actions[trained_agent_id] = env.action_space[trained_agent_id].sample()

            # Then, we use the policy to fill the RB
            else:
                actions[trained_agent_id] = agent.choose_action(state[trained_agent_id])

            for amr in env.agents:
                if amr.id != trained_agent_id:
                    actions[amr.id] = None

            next_state, rewards, terminated, truncated, info = env.step(actions)

            agent.store_transition(
                state[trained_agent_id],
                actions[trained_agent_id],
                rewards[trained_agent_id],
                next_state[trained_agent_id],
                env.dones[trained_agent_id],
            )

            score += rewards[trained_agent_id]
            state = next_state
            total_steps += 1

            if total_steps >= warmup_steps and total_steps % update_frequency == 0:
                agent.learn()

            if env.debug:  # Store metrics for debug logs
                for agent_id, info in info.items():
                    debug.append(
                        {
                            "experiment": env.name,
                            "episode": episode,
                            "step": total_steps,
                            "agent": agent_id,
                            "pos_x": info["pos_x"],
                            "pos_y": info["pos_y"],
                            "distance_to_goal": info["distance_to_goal"],
                            "heading_error": info["heading_error"],
                            "min_obstacle_distance": info["min_obstacle_distance"],
                            "v": info["v"],
                            "omega": info["omega"],
                            "reward": rewards[agent_id],
                            "reward_progress": info["reward_progress"],
                            "reward_rotation": info["reward_rotation"],
                            "reward_safety": info["reward_safety"],
                            "reward_collision": info["reward_collision"],
                            "state": info["state"],
                        }
                    )

        if not env.debug:
            history.append(info)

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        print(
            f"[ {exp.name} ] Step {total_steps:06d}/{n_steps} | Episode {episode:04d} | "
            f"Return = {score:8.3f} "
            f"Average = {avg_score:8.3f} "
            f"Best Average = {best_score:8.3f} "
            f"Success = {info['success_rate']:.0%} "
            f"Time = {info['mean_time_travel']:5.1f} ",
            end="\r",
        )

    print()

    return history, debug, agent
