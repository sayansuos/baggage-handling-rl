import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from configs.config import Experiment
from rl.sac.sac import evaluate_sac, run_sac
from simulator.environment.environment import Environment
from utils.logging import log_debug, log_rewards, log_train
from utils.plotting import (
    plot_animation,
    plot_grid,
    plot_performances,
    plot_rewards,
    plot_velocities,
)


def run_worker(exp: Experiment, worker_id: int) -> tuple[list[dict], int]:
    """
    One process handles multiple episodes on a single environment.
    """

    env = Environment(
        exp.env_config, exp.agent_config, exp.reward_config, exp.name, worker_id
    )

    history = []
    steps_total = 0
    n_episodes = 0

    while steps_total < exp.n_steps:
        n_episodes += 1
        env.reset(seed=1234 + worker_id + steps_total)

        while not env.done() and steps_total < exp.n_steps:
            _, _, _, _, info = env.step()
            steps_total += 1

        history.append(info)

    return history, n_episodes


def run_simulation(experiments: list[Experiment]):
    """
    Run multiple episodes in parallel.
    """

    start = time.perf_counter()
    print(f"\n[ SIMULATION ] {len(experiments)} experiment(s)\n")

    total_eps = 0
    total_steps = 0

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        histories = []

        tasks = [(exp, worker_id) for worker_id, exp in enumerate(experiments)]
        n_processes = min(cpu_count(), len(tasks))

        with Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.starmap(run_worker, tasks),
                    total=len(tasks),
                    desc=f"Simulation -- {exp.name}",
                )
            )

        for history, n_episodes in results:
            histories.extend(history)
            total_eps += n_episodes

        total_steps += exp.n_steps

        df_train = log_train(histories, "logs/simulation", exp.name)
        df_rewards = log_rewards(histories, "logs/simulation", exp.name)

        plot_performances(df_train, "figures/simulation", exp.name)
        plot_velocities(df_train, "figures/simulation", exp.name)
        plot_rewards(df_rewards, "figures/simulation", exp.name)

        print(f"[ DONE ] {exp.name}")

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] {total_eps:,} episode(s) | "
        f"{total_steps:,} step(s) | {duration:.1f}s\n"
    )


def run_test(experiments: list[Experiment], render: bool = False):
    """
    Run test episodes until exp.n_steps is reached.
    """

    print(f"\n[ TEST ] {len(experiments)} experiment(s)\n")

    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = None, None

    debugs = []

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        env = Environment(
            env_config=exp.env_config,
            agent_config=exp.agent_config,
            reward_config=exp.reward_config,
            name=exp.name,
            env_id=1,
            debug=True,
        )

        env.reset(seed=1234)
        plot_grid(env.grid_map.grid, "figures/test", exp.name)

        if render:
            env.render(ax=ax)
            plt.pause(0.001)

        while not env.done():
            _, rewards, _, _, infos = env.step()

            for agent_id, agent_info in infos.items():
                debugs.append(
                    {
                        "experiment": env.name,
                        "episode": env.episode,
                        "step": env.step_count,
                        "agent": agent_id,
                        "pos_x": agent_info["pos_x"],
                        "pos_y": agent_info["pos_y"],
                        "distance_to_goal": agent_info["distance_to_goal"],
                        "heading_error": agent_info["heading_error"],
                        "min_obstacle_distance": agent_info["min_obstacle_distance"],
                        "v": agent_info["v"],
                        "omega": agent_info["omega"],
                        "reward": rewards[agent_id],
                        "reward_progress": agent_info["reward_progress"],
                        "reward_rotation": agent_info["reward_rotation"],
                        "reward_safety": agent_info["reward_safety"],
                        "reward_collision": agent_info["reward_collision"],
                        "state": agent_info["state"],
                    }
                )

            if render:
                env.render(ax=ax)
                plt.pause(0.001)

        log_debug(debugs, "logs/test", exp.name)

        print(f"[ DONE ] {exp.name}")

    if render:
        plt.ioff()
        plt.close(fig)

    print("\n[ TEST ] completed\n")


def run_animation(
    experiments: list[Experiment],
    fps: int = 20,
):
    """
    Run one episode per experiment and save the animation.
    """

    print(f"\n[ SAVE ] {len(experiments)} experiment(s)\n")

    frames = []

    for exp in experiments:
        print(f"[ START ] {exp.name}")

        env = Environment(exp.env_config, exp.agent_config, exp.reward_config, exp.name)
        env.reset(seed=1234)

        fig, ax = plt.subplots(figsize=(10, 8))

        env.render(ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()).copy())

        while not env.done():
            env.step()
            env.render(ax)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.renderer.buffer_rgba()).copy()
            frames.append(frame)

        print(f"[ DONE ] {exp.name}")

    plt.close(fig)

    plot_animation(frames, "figures/test", "", fps)

    print("\n[ SAVE ] completed\n")


def run_train(experiments: list[Experiment], previous_exp: Experiment | None = None):
    """
    Train a SAC agent on each experiment.
    """

    np.random.seed(1234)

    start = time.perf_counter()
    print(f"\n[ TRAIN ] {len(experiments)} experiment(s)\n")

    for exp in experiments:
        history, debug, _ = run_sac(exp=exp, previous_exp=previous_exp)

        df_train = log_train(history, "logs/train", exp.name)
        log_debug(debug, "logs/train", exp.name)
        df_rewards = log_rewards(history, "logs/train", exp.name)

        plot_performances(df_train, "figures/train", exp.name)
        plot_velocities(df_train, "figures/train", exp.name)
        plot_rewards(df_rewards, "figures/train", exp.name)

        previous_exp = exp

    duration = time.perf_counter() - start

    print(
        f"\n[ SUMMARY ] "
        f"{len(experiments)} experiment(s) completed "
        f"in {duration:.1f}s\n"
    )


def run_evaluation(
    experiments: list[Experiment],
    n_episodes: int = 100,
    n_render: int = 5,
):

    np.random.seed(1234)

    for exp in experiments:

        history, frames = evaluate_sac(
            exp=exp,
            n_episodes=n_episodes,
            n_render=n_render,
            trained_agent_id="agent_1",
        )

        df_train = log_train(history, "logs/eval", exp.name)
        df_rewards = log_rewards(history, "logs/eval", exp.name)

        plot_performances(df_train, "figures/eval", exp.name)
        plot_velocities(df_train, "figures/eval", exp.name)
        plot_rewards(df_rewards, "figures/eval", exp.name)

        if n_render > 0:
            plot_animation(frames, "figures/eval", exp.name, 10)
