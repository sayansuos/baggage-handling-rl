import warnings
import numpy as np

from rl.SAC.agent import SACAgent
from simulator.configs.config import Experiment
from simulator.environment.environment import Environment

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_sac(exp: Experiment, worker_id: int = 1, n_episodes: int = 1000):

    env = Environment(
        exp.env_config,
        exp.agent_config,
        exp.reward_config,
        exp.name,
        worker_id,
    )

    trained_agent_id = "agent_1"

    agent = SACAgent(
        env_name=f"{exp.name}_{worker_id}",
        map_shape=(
            1,
            exp.agent_config.length_view,
            exp.agent_config.length_view,
        ),
        action_space=env.action_space[trained_agent_id],
        tau=0.005,
        alpha=0.2,
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        feature_size=256,
        hidden_size=128,
        mem_size=100_000,
    )

    best_score = -np.inf
    history = []
    metrics = []

    for episode in range(n_episodes):

        state, _ = env.reset()
        done = False
        score = 0.0
        while not env.is_done(trained_agent_id):
            actions = {}
            actions[trained_agent_id] = agent.choose_action(state[trained_agent_id])
            for amr in env.agents:
                if amr.id != trained_agent_id:
                    actions[amr.id] = None
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated[trained_agent_id] or truncated[trained_agent_id]

            agent.store_transition(
                state[trained_agent_id],
                actions[trained_agent_id],
                rewards[trained_agent_id],
                next_state[trained_agent_id],
                done,
            )

            agent.learn()
            score += rewards[trained_agent_id]
            state = next_state

        history.append(score)
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        metrics.append(
            {
                "experiment": exp.name,
                "worker": worker_id,
                "episode": episode + 1,
                "return": score,
                "average_return": avg_score,
                "best_return": best_score,
            }
        )

        print(
            f"[{exp.name}_{worker_id} Episode {episode + 1:04}/{n_episodes}] "
            f"Return = {score:8.3f} "
            f"Average = {avg_score:8.3f} "
            f"Best = {best_score:8.3f}",
            end="\r",
        )

    return history, metrics, best_score, agent
