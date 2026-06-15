import warnings

import numpy as np

from configs.config import Experiment
from rl.sac.agent import SACAgent
from simulator.environment.environment import Environment

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_sac(exp: Experiment, n_episodes: int = 1000):

    env = Environment(
        exp.env_config,
        exp.agent_config,
        exp.reward_config,
        exp.name,
    )

    trained_agent_id = "agent_1"

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

    best_score = -np.inf
    best_mean_time_travel = np.inf
    best_success_rate = 0.0
    scores = []
    mean_time_travels = []
    success_rates = []
    metrics = []

    steps = 0
    start_steps = 500

    for episode in range(n_episodes):

        state, _ = env.reset()
        score = 0.0

        if episode % 10 == 0:
            env.debug = True
        else:
            env.debug = False

        while not env.dones[trained_agent_id]:
            actions = {}
            if steps < start_steps:
                actions[trained_agent_id] = env.action_space[trained_agent_id].sample()
            else:
                actions[trained_agent_id] = agent.choose_action(state[trained_agent_id])
                for amr in env.agents:
                    if amr.id != trained_agent_id:
                        actions[amr.id] = None
                agent.learn()
            next_state, rewards, _, _, info = env.step(actions)

            agent.store_transition(
                state[trained_agent_id],
                actions[trained_agent_id],
                rewards[trained_agent_id],
                next_state[trained_agent_id],
                env.dones[trained_agent_id],
            )

            score += rewards[trained_agent_id]
            state = next_state

        history.append(info)
        steps += env.step_count

        scores.append(score)
        mean_time_travels.append(info["mean_time_travel"])
        success_rates.append(info["success_rate"])
        avg_score = np.mean(scores[-100:])
        avg_mean_time_travel = np.mean(mean_time_travels[-100:])
        avg_success_rate = np.mean(success_rates[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()
        if avg_mean_time_travel < best_mean_time_travel:
            best_mean_time_travel = avg_mean_time_travel
        if avg_success_rate > best_success_rate:
            best_success_rate = avg_success_rate

        metrics.append(
            {
                "experiment": exp.name,
                "episode": episode + 1,
                "return": score,
                "average_return": avg_score,
                "best_return": best_score,
                "mean_time_travel": info["mean_time_travel"],
                "average_mean_time_travel": avg_mean_time_travel,
                "best_mean_time_travel": best_mean_time_travel,
                "success_rate": info["success_rate"],
                "average_success_rate": avg_success_rate,
                "best_success_rate": best_success_rate,
            }
        )

        print(
            f"[{exp.name} Episode {episode + 1:04}/{n_episodes}] "
            f"Return = {score:8.3f} "
            f"Average = {avg_score:8.3f} "
            f"Best = {best_score:8.3f}",
            end="\r",
        )

    return history, scores, metrics, best_score, agent, steps
